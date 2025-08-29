# Copyright 2024 The Sigstore Authors
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable, Iterable
import pathlib
from typing import TypeAlias
import torch

from .manifest import manifest
from .serialization import serialization, serialize_by_file, serialize_by_state
from .signature import verifying
from .signing import signing
from .hashing import hashing, file, state
from .hashing.topology import *

from cuda.bindings import driver
from .cuda.compiler import checkCudaErrors

import time

PayloadGeneratorFunc: TypeAlias = Callable[
    [manifest.Manifest], signing.SigningPayload
]

def build_serializer(item, hashAlgo: HashAlgo, topology: Topology, workflow: Workflow):
    if isinstance(item, pathlib.Path):
        def hasher_factory(item) -> hashing.HashEngine:
            return file.SimpleFileHasher(
                file=item, content_hasher=hashAlgo.value[2]())
        serializer = serialize_by_file.ManifestSerializer(
            file_hasher_factory=hasher_factory)

    elif isinstance(item, torch.nn.Module):
        device = 'gpu' if next(item.parameters()).is_cuda else 'cpu'
        hasher = get_model_hasher(hashAlgo, topology, workflow, device)
        def hasher_factory(item) -> hashing.HashEngine:
            return state.SimpleStateHasher(state=item, content_hasher=hasher)
        serializer = serialize_by_state.ManifestSerializer(
            state_hasher_factory=hasher_factory)

    elif isinstance(item, dict):
        serializer = serialize_by_state.ManifestSerializer(
            state_hasher_factory=None)

    return serializer

def sign(
    item: pathlib.Path | torch.nn.Module | dict,
    signer: signing.Signer,
    payload_generator: PayloadGeneratorFunc,
    hashAlgo: HashAlgo,
    topology: Topology,
    workflow: Workflow,
    ignore_paths: Iterable[pathlib.Path] = frozenset(),
) -> Iterable[signing.Signature]:
    """Provides a wrapper function for the steps necessary to sign a model.

    Args:
        item: the input to be hashed
        signer: the signer to be used.
        payload_generator: funtion to generate the manifest.
        serializer: the serializer to be used for the model.
        ignore_paths: paths that should be ignored during serialization.
          Defaults to an empty set.

    Returns:
        The model's signature.
    """

    if isinstance(item, dict):
        stmnts = []
        hashes = []
        serializer = build_serializer(item, None, None, None)
        for identity, (digest, trueHash) in item.items():
            manifestItem = manifest.StateManifestItem(
                state=identity, digest=digest)
            manif = serializer._build_manifest([manifestItem])
            stmnts.append(payload_generator(manif))
            hashes.append([trueHash])
    else:
        serializer = build_serializer(item, hashAlgo, topology, workflow)
        start = time.perf_counter()
        if isinstance(item, torch.nn.Module):
            item = item.state_dict()
        manif = serializer.serialize(item, ignore_paths=ignore_paths)
        stmnts = [payload_generator(manif)]
        hashes = [serializer.trueHashes if hasattr(serializer, 'trueHashes') else None]
        end = time.perf_counter()
        print(f'[Trainer] Model hashing runtime: {1000*(end-start):.2f} ms')
    return signer.sign(stmnts, hashes)


def verify(
    sig: signing.Signature | Iterable[signing.Signature],
    verifier: signing.Verifier,
    item: pathlib.Path | torch.nn.Module | dict,
    ignore_paths: Iterable[pathlib.Path] = frozenset(),
):
    """Provides a simple wrapper to verify models.

    Args:
        sig: the signature to be verified.
        verifier: the verifier to verify the signature.
        model_path: the path to the model to compare manifests.
        serializer: the serializer used to generate the local manifest.
        ignore_paths: paths that should be ignored during serialization.
          Defaults to an empty set.

    Raises:
        verifying.VerificationError: on any verification error.
    """
    sigs = sig if isinstance(sig, list) else [sig]

    peer_manifests, algo = verifier.verify(sigs)
    # figure out hashing algorithm from signature
    topology, workflow, hashAlgo = algo.split('-')
    serializer = build_serializer(item, HashAlgo[hashAlgo], Topology[topology], Workflow[workflow])

    local_manifests = []

    if isinstance(item, dict):
        for identity in [next(iter(m._item_to_digest.keys())) for m in peer_manifests]:
            digest, trueHash = item[identity]
            checkCudaErrors(driver.cuMemcpyDtoH(digest.digest_value, trueHash, digest.digest_size))
            manifestItem = manifest.StateManifestItem(state=identity, digest=digest)
            local_manifests.append(serializer._build_manifest([manifestItem]))
    else:
        if isinstance(item, torch.nn.Module):
            item = item.state_dict()
        start = time.perf_counter()
        manifestItem = serializer.serialize(item, ignore_paths=ignore_paths)
        if hasattr(serializer, 'trueHashes'):
            for digest, trueHash in zip(manifestItem._item_to_digest.values(), serializer.trueHashes):
                checkCudaErrors(driver.cuMemcpyDtoH(digest.digest_value, trueHash, digest.digest_size))
        local_manifests.append(manifestItem)
        end = time.perf_counter()
        print(f'[Inferencer] Model hashing runtime: {1000*(end-start):.2f} ms')

    for peer, local in zip(peer_manifests, local_manifests):
        if peer != local:
            raise verifying.VerificationError(f'the manifests do not match')
