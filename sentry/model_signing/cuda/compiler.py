from cuda.bindings import driver, nvrtc, runtime
import numpy as np
import os

def _cudaGetErrorEnum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))


def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(result[0].value,
            _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]
    

def rtcompile(srcPath: str, function_names: list[str], flags=[]):
    cuDevice = checkCudaErrors(runtime.cudaGetDevice())
    ctx = checkCudaErrors(driver.cuCtxGetCurrent())
    if repr(ctx) == '<CUcontext 0x0>':
        ctx = checkCudaErrors(driver.cuCtxCreate(0, cuDevice))
    # get device arch
    major = checkCudaErrors(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        cuDevice))
    minor = checkCudaErrors(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        cuDevice))
    arch_arg = bytes(f'--gpu-architecture=compute_{major}{minor}', 'ascii')

    inclPath = os.path.dirname(srcPath)
    opts = [
            b'--fmad=false', arch_arg,
            b'-I' + inclPath.encode(),
            b'-I/usr/local/cuda/include',
        ]
    for flag in flags:
        opts.append(bytes(f'-D{flag}', 'ascii'))

    with open(srcPath, 'r') as f:
        code = f.read()
    # parse cuda code from file
    prog = checkCudaErrors(nvrtc.nvrtcCreateProgram(str.encode(code), 
        bytes(srcPath, 'utf-8'), 0, [], []))
    
    # compile code into program and extract ptx
    err = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if err[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        logSize = checkCudaErrors(nvrtc.nvrtcGetProgramLogSize(prog))
        log = bytes(logSize)
        checkCudaErrors(nvrtc.nvrtcGetProgramLog(prog, log))
        print(log.decode("utf-8"), flush=True)
        checkCudaErrors(err)

    ptxSize = checkCudaErrors(nvrtc.nvrtcGetPTXSize(prog))
    ptx = b' ' * ptxSize
    checkCudaErrors(nvrtc.nvrtcGetPTX(prog, ptx))
    ptx = np.char.array(ptx)
    # obtain global functions as entrypoints into gpu
    module = checkCudaErrors(driver.cuModuleLoadData(ptx.ctypes.data))
    funcs = []
    for func in function_names:
        funcs.append(checkCudaErrors(driver.cuModuleGetFunction(module,
            bytes(f'{func}', 'utf-8'))))
    return ctx, funcs
