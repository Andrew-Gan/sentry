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

# NOTE: item type used to infer hashing method
# item type         |   ML artifact |   format              |   device
# str               |   model       |   state files         |   cpu
# torch.nn.Module   |   model       |   loaded into torch   |   cpu or gpu
# dict              |   dataset     |   per-sample digests  |   gpu

def sign_model(item: str | torch.nn.Module, hashAlgo=HashAlgo.SHA256,
    topology=Topology.MERKLE, workflow=Workflow.INPLACE, sig_filename='model.sig'):

    # check validity of configuration
    if topology == Topology.LATTICE and hashAlgo != HashAlgo.BLAKE2XB:
        raise RuntimeError('LATTICE only supported with BLAKE2XB')
    if workflow == Workflow.LAYERED_SORTED and topology != Topology.LATTICE:
        raise RuntimeError('LAYERED_SORTED only supported with Topology.LATTICE')

    args = sign._arguments()
    if isinstance(item, str):
        signer = sign._get_payload_signer(args, 'cpu')
        item = pathlib.Path(item)
    elif isinstance(item, torch.nn.Module):
        dev = 'gpu' if next(item.parameters()).is_cuda else 'cpu'
        signer = sign._get_payload_signer(args, dev)
    else:
        raise TypeError('item is neither str nor torch module')

    payload_generator = in_toto.DigestsIntotoPayload.from_manifest
    sig = model.sign(item, signer, payload_generator, hashAlgo, topology,
        workflow, ignore_paths=[args.sig_out])[0]
    sig.write(args.sig_out / pathlib.Path(sig_filename))


def sign_dataset(item: dict):
    args = sign._arguments()
    signer = sign._get_payload_signer(args, 'gpu', len(item))
    payload_generator = in_toto.DigestsIntotoPayload.from_manifest
    sigs = model.sign(item, signer, payload_generator, None, None, None)
    for src, sig in zip(item.keys(), sigs):
        sig.write(args.sig_out / pathlib.Path(f'dataset_{src}.sig'))


def verify_model(item: str | torch.nn.Module, sig_filename='model.sig'):
    args = verify._arguments()
    args.sig_path = args.sig_path / pathlib.Path(sig_filename)
    sig = verify._get_signature(args)
    if isinstance(item, str):
        verifier = verify._get_verifier(args, 'cpu')
        item = pathlib.Path(item)
    elif isinstance(item, torch.nn.Module):
        verifier = verify._get_verifier(args, 'gpu')
    else:
        raise TypeError('item is neither str nor torch module')

    try:
        model.verify(
            sig=sig,
            item=item,
            verifier=verifier,
            ignore_paths=[args.sig_path],
        )
    except verifying.VerificationError as err:
        log.error(f"verification failed: {err}")

    log.info("all checks passed")


def verify_dataset(item: dict):
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
        )
    except verifying.VerificationError as err:
        log.error(f"verification failed: {err}")

    log.info("all checks passed")
