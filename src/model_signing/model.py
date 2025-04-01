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

PayloadGeneratorFunc: TypeAlias = Callable[
    [manifest.Manifest], signing.SigningPayload
]


import time


def sign(
    item,
    signer: signing.Signer,
    payload_generator: PayloadGeneratorFunc,
    serializer: serialization.Serializer,
    ignore_paths: Iterable[pathlib.Path] = frozenset(),
) -> signing.Signature:
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
    t0 = time.monotonic()
    manif, _ = serializer.serialize(item, ignore_paths=ignore_paths)
    t1 = time.monotonic()
    payload = payload_generator(manif)
    sig = signer.sign(payload)
    t2 = time.monotonic()
    print(f'{(t1-t0)*1000:.2f}, {(t2-t1)*1000:.2f}')
    return sig


def sign_hash(
    hashes: Iterable[hashing.Digest],
    signer: signing.Signer,
    payload_generator: PayloadGeneratorFunc,
    serializer: serialization.Serializer,
) -> signing.Signature:

    payload = []
    for i, hash in enumerate(hashes):
        manifestItem = manifest.StateManifestItem(state=i, digest=hash)
        manif = serializer._build_manifest([manifestItem])
        payload.append(payload_generator(manif))
    sigs = signer.sign(payload)
    return sigs


def verify(
    sig: signing.Signature,
    verifier: signing.Verifier,
    model_path: pathlib.Path,
    serializer: serialization.Serializer,
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
    peer_manifest = verifier.verify(sig)
    local_manifest = serializer.serialize(model_path, ignore_paths=ignore_paths)
    if peer_manifest != local_manifest:
        raise verifying.VerificationError("the manifests do not match")
