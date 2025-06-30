import cupy
import cupy.cuda.nccl as nccl


def to_nccl_data_type(cupy_array):
    nccl_type_dict = {
        "ncclInt8": 0,
        "ncclChar": 0,
        "ncclUint8": 1,
        "ncclInt32": 2,
        "ncclInt": 2,
        "ncclUint32": 3,
        "ncclInt64": 4,
        "ncclUint64": 5,
        "ncclFloat16": 6,
        "ncclHalf": 6,
        "ncclFloat32": 7,
        "ncclFloat": 7,
        "ncclFloat64": 8,
        "ncclDouble": 8,
    }

    return nccl_type_dict["nccl" + str(cupy_array.dtype).capitalize()]


class Communicator:
    def __init__(self, gpu_id_list=None):

        self.size = 1
        self.rank = 0
        self.local_size = 1
        self.local_rank = 0
        self.is_main = True

        try:
            from mpi4py import MPI

            self.world = MPI.COMM_WORLD

            self.is_main = self.world.rank == 0

            processor_name = MPI.Get_processor_name()
            rank = self.world.rank

            host_names = self.world.gather(processor_name)

            # This removes redundant host names. Also the order can be random
            # if the removal is operated individually
            if self.is_main:
                host_names = list(set(host_names))

            host_names = self.world.bcast(host_names)
            color = host_names.index(processor_name)
            self.local = self.world.Split(color, rank)
            self.rank = rank
            self.size = self.world.size
            self.local_rank = self.local.rank

        except:
            self.world = None
            self.local = None

        try:
            unique_id = nccl.get_unique_id()
            unique_id = self.world.bcast(unique_id)

            n_gpu = cupy.cuda.runtime.getDeviceCount()
            if gpu_id_list is None:
                gpu_id_list = range(n_gpu)

            cupy.cuda.Device(gpu_id_list[self.local_rank]).use()

            if self.local_size > n_gpu:
                raise Exception(
                    "the size of local processes exceeds allocable GPU devices"
                )

            self.gpu = nccl.NcclCommunicator(self.size, unique_id, self.rank)

        except:
            self.gpu = None

    def reduce(self, cupy_array: cupy.ndarray, in_place=False):
        nccl_sum_type = 0
        default_stream = 0

        if self.size == 1:
            return cupy_array

        if not in_place:
            result = cupy.ndarray(cupy_array.shape, dtype=cupy_array.dtype)
        else:
            result = cupy_array

        if cupy.iscomplexobj(cupy_array):
            self.gpu.allReduce(
                cupy_array.real.data.ptr,
                result.real.data.ptr,
                cupy_array.size,
                to_nccl_data_type(cupy_array.real),
                nccl_sum_type,
                default_stream,
            )
            self.gpu.allReduce(
                cupy_array.imag.data.ptr,
                result.imag.data.ptr,
                cupy_array.size,
                to_nccl_data_type(cupy_array.imag),
                nccl_sum_type,
                default_stream,
            )

        else:
            self.gpu.allReduce(
                cupy_array.data.ptr,
                result.data.ptr,
                cupy_array.size,
                to_nccl_data_type(cupy_array),
                nccl_sum_type,
                default_stream,
            )

        return result

    def gather(self, cupy_array: cupy.ndarray):

        default_stream = 0
        if self.size == 1:
            return cupy_array
        
        shape = list(cupy_array.shape)
        shape[0] *= self.size

        result = cupy.ndarray(shape, dtype=cupy_array.dtype)

        if cupy.iscomplexobj(cupy_array):
            self.gpu.allGather(
                cupy_array.real.data.ptr,
                result.real.data.ptr,
                cupy_array.size,
                to_nccl_data_type(cupy_array.real),
                default_stream,
            )
            self.gpu.allGather(
                cupy_array.imag.data.ptr,
                result.imag.data.ptr,
                cupy_array.size,
                to_nccl_data_type(cupy_array.imag),
                default_stream,
            )
        else:
            
            
            self.gpu.allGather(
                cupy_array.data.ptr,
                result.data.ptr,
                cupy_array.size,
                to_nccl_data_type(cupy_array),
                default_stream,
            )

        return result


comm = Communicator()

def get_master_print_level(intended_level):
    if comm.is_main:
        return intended_level
    else:
        return 0

    