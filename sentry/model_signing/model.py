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

from .manifest import manifest
from .serialization import serialization
from .signature import verifying
from .signing import signing
from .hashing import hashing
import collections
import pathlib

from cuda.bindings import driver
from .cuda.compiler import checkCudaErrors

PayloadGeneratorFunc: TypeAlias = Callable[
    [manifest.Manifest], signing.SigningPayload
]


def sign(
    item: pathlib.Path | collections.OrderedDict | Iterable[hashing.Digest],
    signer: signing.Signer,
    payload_generator: PayloadGeneratorFunc,
    serializer: serialization.Serializer,
    ignore_paths: Iterable[pathlib.Path] = frozenset(),
    isDigest: bool = False,
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

    if isDigest:
        stmnts = []
        hashes = []
        for src, (digest, trueHash) in item.items():
            manifestItem = manifest.StateManifestItem(state=src, digest=digest)
            manif = serializer._build_manifest([manifestItem])
            stmnts.append(payload_generator(manif))
            hashes.append([trueHash])
    else:
        manif = serializer.serialize(item, ignore_paths=ignore_paths)
        stmnts = [payload_generator(manif)]
        hashes = [serializer.trueHashes if hasattr(serializer, 'trueHashes') else None]
    return signer.sign(stmnts, hashes)


def verify(
    sig: signing.Signature | Iterable[signing.Signature],
    verifier: signing.Verifier,
    item: pathlib.Path | collections.OrderedDict | Iterable[hashing.Digest],
    serializer: serialization.Serializer,
    ignore_paths: Iterable[pathlib.Path] = frozenset(),
    isDigest: bool = False,
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
    items = item if isinstance(item, list) else [item]
    sigs = sig if isinstance(sig, list) else [sig]

    peer_manifests = verifier.verify(sigs)
    local_manifests = []

    if isDigest:
        for src, (digest, trueHash) in item.items():
            manifest = manifest.StateManifestItem(
                state=src, digest=digest)
            local_manifests.append(serializer._build_manifest([manifest]))
    else:
        manifest = serializer.serialize(item, ignore_paths=ignore_paths)
        if hasattr(serializer, 'trueHashes'):
            for digest, trueHash in zip(manifest._item_to_digest.values(), serializer.trueHashes):
                checkCudaErrors(driver.cuMemcpyDtoH(digest.digest_value, trueHash, 32))
        local_manifests.append(manifest)

    for peer, local in zip(peer_manifests, local_manifests):
        if peer != local:
            raise verifying.VerificationError(f'the manifests do not match')
