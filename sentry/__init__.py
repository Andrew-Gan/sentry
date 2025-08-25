from . import sign
from . import verify
from .model_signing import model
from .model_signing.hashing.topology import *
from .model_signing.signature import verifying
from .model_signing.signing import in_toto

import pathlib
import torch
import logging
import os

log = logging.getLogger(__name__)

def sign_model(item, hashAlgo=HashAlgo.SHA256, topology=Topology.MERKLE_INPLACE):
    args = sign._arguments()
    if isinstance(item, str):
        signer, serializer = sign.build(hashAlgo, topology, InputType.FILE, 'cpu')
        item = pathlib.Path(item)
    elif isinstance(item, torch.nn.Module):
        dev = 'gpu' if next(item.parameters()).is_cuda else 'cpu'
        signer, serializer = sign.build(hashAlgo, topology, InputType.MODULE, dev)
        item = item.state_dict()
    else:
        raise TypeError('item is neither str nor torch module')

    sig = model.sign(
        item=item,
        signer=signer,
        payload_generator=in_toto.DigestsIntotoPayload.from_manifest,
        serializer=serializer,
        ignore_paths=[args.sig_out],
    )[0]
    sig.write(args.sig_out / pathlib.Path('model.sig'))


def sign_dataset(item: collections.OrderedDict, hashAlgo=HashAlgo.BLAKE2XB, topology=Topology.LATTICE):
    signer, serializer = sign.build(hashAlgo, topology, InputType.DIGEST, 'gpu', len(item))
    args = sign._arguments()
    sigs = model.sign(
        item=item,
        signer=signer,
        payload_generator=in_toto.DigestsIntotoPayload.from_manifest,
        serializer=serializer,
        isDigest=True,
    )
    for src, sig in zip(item.keys(), sigs):
        sig.write(args.sig_out / pathlib.Path(f'dataset_{src}.sig'))


def verify_model(item, hashAlgo=HashAlgo.SHA256, topology=Topology.MERKLE_INPLACE):
    args = verify._arguments()
    args.sig_path = args.sig_path / pathlib.Path('model.sig')
    sig = verify._get_signature(args)
    if isinstance(item, str):
        verifier, serializer = verify.build(hashAlgo, topology, InputType.FILE, 'cpu')
        item = pathlib.Path(item)
    elif isinstance(item, torch.nn.Module):
        dev = 'gpu' if next(item.parameters()).is_cuda else 'cpu'
        verifier, serializer = verify.build(hashAlgo, topology, InputType.MODULE, dev)
        item = item.state_dict()
    else:
        raise TypeError('item is neither str nor torch module')

    try:
        model.verify(
            sig=sig,
            item=item,
            verifier=verifier,
            serializer=serializer,
            ignore_paths=[args.sig_path],
        )
    except verifying.VerificationError as err:
        log.error(f"verification failed: {err}")

    log.info("all checks passed")


def verify_dataset(item: collections.OrderedDict, hashAlgo=HashAlgo.BLAKE2XB, topology=Topology.LATTICE):
    verifier, serializer = verify.build(hashAlgo, topology, InputType.DIGEST, 'gpu', len(item))
    args = verify._arguments()
    sig_path = args.sig_path
    sig = []

    for filename in os.listdir(sig_path):
        if filename.startswith('dataset_') and filename.endswith('.sig'):
            args.sig_path = sig_path / pathlib.Path(filename)
            sig.append(verify._get_signature(args))
    try:
        model.verify(
            sig=sig,
            item=item,
            verifier=verifier,
            serializer=serializer,
            ignore_paths=[args.sig_path],
            isDigest=True,
        )
    except verifying.VerificationError as err:
        log.error(f"verification failed: {err}")

    log.info("all checks passed")
