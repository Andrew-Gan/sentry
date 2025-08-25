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
        signer = sign._get_payload_signer(args, 'cpu')
        serializer = model.build_serializer(hashAlgo, topology, InputType.FILE, 'cpu')
        item = pathlib.Path(item)
        inputType = InputType.FILE
    elif isinstance(item, torch.nn.Module):
        dev = 'gpu' if next(item.parameters()).is_cuda else 'cpu'
        signer = sign._get_payload_signer(args, dev)
        serializer = model.build_serializer(hashAlgo, topology, InputType.MODULE, dev)
        item = item.state_dict()
        inputType = InputType.MODULE
    else:
        raise TypeError('item is neither str nor torch module')

    sig = model.sign(
        item=item,
        signer=signer,
        payload_generator=in_toto.DigestsIntotoPayload.from_manifest,
        serializer=serializer,
        inputType=inputType,
        ignore_paths=[args.sig_out],
    )[0]
    sig.write(args.sig_out / pathlib.Path('model.sig'))


def sign_dataset(item: collections.OrderedDict, hashAlgo=HashAlgo.BLAKE2XB, topology=Topology.LATTICE):
    args = sign._arguments()
    signer = sign._get_payload_signer(args, 'gpu', len(item))
    serializer = model.build_serializer(hashAlgo, topology, InputType.DIGEST, 'gpu')
    sigs = model.sign(
        item=item,
        signer=signer,
        payload_generator=in_toto.DigestsIntotoPayload.from_manifest,
        serializer=serializer,
        inputType=InputType.DIGEST,
    )
    for src, sig in zip(item.keys(), sigs):
        sig.write(args.sig_out / pathlib.Path(f'dataset_{src}.sig'))


def verify_model(item: str | torch.nn.Module):
    args = verify._arguments()
    args.sig_path = args.sig_path / pathlib.Path('model.sig')
    sig = verify._get_signature(args)
    if isinstance(item, str):
        verifier = verify._get_verifier(args, 'cpu')
        item = pathlib.Path(item)
        inputType = InputType.FILE
    elif isinstance(item, torch.nn.Module):
        dev = 'gpu' if next(item.parameters()).is_cuda else 'cpu'
        verifier = verify._get_verifier(args, 'gpu')
        item = item.state_dict()
        inputType = InputType.MODULE
    else:
        raise TypeError('item is neither str nor torch module')

    try:
        model.verify(
            sig=sig,
            item=item,
            verifier=verifier,
            ignore_paths=[args.sig_path],
            inputType=inputType
        )
    except verifying.VerificationError as err:
        log.error(f"verification failed: {err}")

    log.info("all checks passed")


def verify_dataset(item: collections.OrderedDict):
    args = verify._arguments()
    verifier = verify._get_verifier(args, 'gpu', len(item))
    sig = []
    sig_path = args.sig_path

    for filename in os.listdir(args.sig_path):
        if filename.startswith('dataset_') and filename.endswith('.sig'):
            args.sig_path = sig_path / pathlib.Path(filename)
            sig.append(verify._get_signature(args))
    try:
        model.verify(
            sig=sig,
            item=item,
            verifier=verifier,
            ignore_paths=[args.sig_path],
            inputType=InputType.DIGEST
        )
    except verifying.VerificationError as err:
        log.error(f"verification failed: {err}")

    log.info("all checks passed")
