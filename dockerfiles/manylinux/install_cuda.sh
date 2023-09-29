Last login: Thu Sep 28 23:56:06 on ttys006

The default interactive shell is now zsh.
To update your account to use zsh, please run `chsh -s /bin/zsh`.
For more details, please visit https://support.apple.com/kb/HT208050.
C02G8BSLMD6R:manylinux bytedance$ 
C02G8BSLMD6R:manylinux bytedance$ 
C02G8BSLMD6R:manylinux bytedance$ ls
Dockerfile		Dockerfile~		README			build-docker.sh		build-wheels.sh		build-wheels.sh~	install_cuda.sh		install_cuda.sh~
C02G8BSLMD6R:manylinux bytedance$ docker build .
[+] Building 82.9s (12/15)                                                                                                                                                                                                                                    
 => [internal] load build definition from Dockerfile                                                                                                                                                                                                     0.0s
 => => transferring dockerfile: 1.42kB                                                                                                                                                                                                                   0.0s
 => [internal] load .dockerignore                                                                                                                                                                                                                        0.0s
 => => transferring context: 2B                                                                                                                                                                                                                          0.0s
 => [internal] load metadata for quay.io/pypa/manylinux2014_x86_64:2023-09-24-36b93e4                                                                                                                                                                    0.4s
 => [ 1/11] FROM quay.io/pypa/manylinux2014_x86_64:2023-09-24-36b93e4@sha256:07145d872015f593222ac57874e0ef14f3f4bde5b391ff4d1be8979c28170d7e                                                                                                            0.0s
 => [internal] load build context                                                                                                                                                                                                                        0.0s
 => => transferring context: 37B                                                                                                                                                                                                                         0.0s
 => CACHED [ 2/11] RUN yum install -y wget curl perl util-linux xz bzip2 git patch which perl zlib-devel                                                                                                                                                 0.0s
 => CACHED [ 3/11] RUN yum install -y yum-utils centos-release-scl                                                                                                                                                                                       0.0s
 => [ 4/11] RUN rm -rf /opt/python/cp26-cp26m /opt/_internal/cpython-2.6.9-ucs2                                                                                                                                                                          0.3s
 => [ 5/11] RUN rm -rf /opt/python/cp26-cp26mu /opt/_internal/cpython-2.6.9-ucs4                                                                                                                                                                         0.3s
 => [ 6/11] RUN rm -rf /opt/python/cp33-cp33m /opt/_internal/cpython-3.3.6                                                                                                                                                                               0.3s
 => [ 7/11] RUN rm -rf /opt/python/cp34-cp34m /opt/_internal/cpython-3.4.6                                                                                                                                                                               0.3s
 => [ 8/11] ADD install_cuda.sh install_cuda.sh                                                                                                                                                                                                          0.0s
 => [ 9/11] RUN bash ./install_cuda.sh 11.8 && rm install_cuda.sh                                                                                                                                                                                       81.2s
 => => # + case "$1" in                                                                                                                                                                                                                                      
 => => # + install_118                                                                                                                                                                                                                                       
 => => # + echo 'Installing CUDA 11.8 and cuDNN 8.7 and NCCL 2.15 and cutensor 1.6.1.5'                                                                                                                                                                      
 => => # Installing CUDA 11.8 and cuDNN 8.7 and NCCL 2.15 and cutensor 1.6.1.5                                                                                                                                                                               
 => => # + rm -rf /usr/local/cuda-11.8 /usr/local/cuda                                                                                                                                                                                                       
 => => # + wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run                                                                                                                                
^CERRO[0083] got 3 SIGTERM/SIGINTs, forcing shutdown      
C02G8BSLMD6R:manylinux bytedance$ 
C02G8BSLMD6R:manylinux bytedance$ 
C02G8BSLMD6R:manylinux bytedance$ 
C02G8BSLMD6R:manylinux bytedance$ docker build .
[+] Building 546.5s (13/16)                                                                                                                                                                                                                                   
 => [internal] load build definition from Dockerfile                                                                                                                                                                                                     0.0s
 => => transferring dockerfile: 1.45kB                                                                                                                                                                                                                   0.0s
 => [internal] load .dockerignore                                                                                                                                                                                                                        0.0s
 => => transferring context: 2B                                                                                                                                                                                                                          0.0s
 => [internal] load metadata for quay.io/pypa/manylinux2014_x86_64:2023-09-24-36b93e4                                                                                                                                                                    0.3s
 => [internal] load build context                                                                                                                                                                                                                        0.0s
 => => transferring context: 37B                                                                                                                                                                                                                         0.0s
 => [ 1/12] FROM quay.io/pypa/manylinux2014_x86_64:2023-09-24-36b93e4@sha256:07145d872015f593222ac57874e0ef14f3f4bde5b391ff4d1be8979c28170d7e                                                                                                            0.0s
 => CACHED [ 2/12] RUN yum install -y wget curl perl util-linux xz bzip2 git patch which perl zlib-devel                                                                                                                                                 0.0s
 => CACHED [ 3/12] RUN yum install -y yum-utils centos-release-scl                                                                                                                                                                                       0.0s
 => CACHED [ 4/12] RUN rm -rf /opt/python/cp26-cp26m /opt/_internal/cpython-2.6.9-ucs2                                                                                                                                                                   0.0s
 => CACHED [ 5/12] RUN rm -rf /opt/python/cp26-cp26mu /opt/_internal/cpython-2.6.9-ucs4                                                                                                                                                                  0.0s
 => CACHED [ 6/12] RUN rm -rf /opt/python/cp33-cp33m /opt/_internal/cpython-3.3.6                                                                                                                                                                        0.0s
 => CACHED [ 7/12] RUN rm -rf /opt/python/cp34-cp34m /opt/_internal/cpython-3.4.6                                                                                                                                                                        0.0s
 => CACHED [ 8/12] ADD install_cuda.sh install_cuda.sh                                                                                                                                                                                                   0.0s
 => ERROR [ 9/12] RUN bash ./install_cuda.sh 11.8 && rm install_cuda.sh                                                                                                                                                                                546.0s
------                                                                                                                                                                                                                                                        
 > [ 9/12] RUN bash ./install_cuda.sh 11.8 && rm install_cuda.sh:                                                                                                                                                                                             
#13 0.241 + test 1 -gt 0                                                                                                                                                                                                                                      
#13 0.241 + case "$1" in                                                                                                                                                                                                                                      
#13 0.241 + install_118                                                                                                                                                                                                                                       
#13 0.241 + echo 'Installing CUDA 11.8 and cuDNN 8.7 and NCCL 2.15 and cutensor 1.6.1.5'                                                                                                                                                                      
#13 0.241 + rm -rf /usr/local/cuda-11.8 /usr/local/cuda
#13 0.241 Installing CUDA 11.8 and cuDNN 8.7 and NCCL 2.15 and cutensor 1.6.1.5
#13 0.243 + wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
#13 460.4 + chmod +x cuda_11.8.0_520.61.05_linux.run
#13 460.4 + ./cuda_11.8.0_520.61.05_linux.run --toolkit --silent
#13 544.7 terminate called after throwing an instance of 'boost::filesystem::filesystem_error'
#13 544.7   what():  boost::filesystem::copy_file: No space left on device: "./builds/cuda_cccl/targets/x86_64-linux/include/cuda/std/detail/libcxx/include/random", "/usr/local/cuda-11.8/targets/x86_64-linux/include/cuda/std/detail/libcxx/include/random"
#13 544.7 ./cuda_11.8.0_520.61.05_linux.run: line 524:    42 Aborted                 ./cuda-installer --toolkit --silent
------
executor failed running [/bin/sh -c bash ./install_cuda.sh ${BASE_CUDA_VERSION} && rm install_cuda.sh]: exit code: 134
C02G8BSLMD6R:manylinux bytedance$ docker ps
CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
C02G8BSLMD6R:manylinux bytedance$ docker ps -a
CONTAINER ID   IMAGE               COMMAND       CREATED         STATUS                       PORTS     NAMES
df3ecb8542ee   gpu4pyscf           "/bin/bash"   10 months ago   Exited (1) 10 months ago               pensive_wilbur
33ebfcfc85d8   gpu4pyscf           "/bin/bash"   10 months ago   Exited (129) 10 months ago             vigilant_murdock
d9dd0cb8ea4a   gpu4pyscf           "bash"        10 months ago   Exited (129) 10 months ago             zealous_panini
98b3d83cbeaa   cupy/cupy:v11.3.0   "/bin/bash"   10 months ago   Exited (1) 10 months ago               exciting_agnesi
c4f9fbbb7ce1   cupy/cupy:v11.3.0   "/bin/bash"   10 months ago   Exited (0) 10 months ago               optimistic_rosalind
20ac1fb6f972   b533a80e387e        "/bin/bash"   10 months ago   Exited (1) 10 months ago               relaxed_tu
C02G8BSLMD6R:manylinux bytedance$ docker rm 98b3d83cbeaa
98b3d83cbeaa
C02G8BSLMD6R:manylinux bytedance$ docker rm 20ac1fb6f972
20ac1fb6f972
C02G8BSLMD6R:manylinux bytedance$ docker rm 20ac1fb6f972d9dd0cb8ea4a
C02G8BSLMD6R:manylinux bytedance$ 
C02G8BSLMD6R:manylinux bytedance$ docker rm d9dd0cb8ea4a
d9dd0cb8ea4a
C02G8BSLMD6R:manylinux bytedance$ 
C02G8BSLMD6R:manylinux bytedance$ 
C02G8BSLMD6R:manylinux bytedance$ ls
Dockerfile		Dockerfile~		README			build-docker.sh		build-wheels.sh		build-wheels.sh~	install_cuda.sh		install_cuda.sh~
C02G8BSLMD6R:manylinux bytedance$ docker ps -a
CONTAINER ID   IMAGE               COMMAND       CREATED         STATUS                       PORTS     NAMES
df3ecb8542ee   gpu4pyscf           "/bin/bash"   10 months ago   Exited (1) 10 months ago               pensive_wilbur
33ebfcfc85d8   gpu4pyscf           "/bin/bash"   10 months ago   Exited (129) 10 months ago             vigilant_murdock
c4f9fbbb7ce1   cupy/cupy:v11.3.0   "/bin/bash"   10 months ago   Exited (0) 10 months ago               optimistic_rosalind
C02G8BSLMD6R:manylinux bytedance$ docker rm c4f9fbbb7ce1
c4f9fbbb7ce1
C02G8BSLMD6R:manylinux bytedance$ docker rm 33ebfcfc85d8
33ebfcfc85d8
C02G8BSLMD6R:manylinux bytedance$ 
C02G8BSLMD6R:manylinux bytedance$ 
C02G8BSLMD6R:manylinux bytedance$ 
C02G8BSLMD6R:manylinux bytedance$ 
C02G8BSLMD6R:manylinux bytedance$ 
C02G8BSLMD6R:manylinux bytedance$ 
C02G8BSLMD6R:manylinux bytedance$ 
C02G8BSLMD6R:manylinux bytedance$ 
C02G8BSLMD6R:manylinux bytedance$ 
C02G8BSLMD6R:manylinux bytedance$ ls
Dockerfile		Dockerfile~		README			build-docker.sh		build-wheels.sh		build-wheels.sh~	install_cuda.sh		install_cuda.sh~
C02G8BSLMD6R:manylinux bytedance$ docker images
REPOSITORY   TAG       IMAGE ID       CREATED         SIZE
<none>       <none>    5681298be998   6 days ago      5.31GB
gpu4pyscf    latest    1e56485ef4b4   10 months ago   6.81GB
cupy/cupy    v11.3.0   b533a80e387e   10 months ago   5.44GB
C02G8BSLMD6R:manylinux bytedance$ docker rm gpu4pyscf
Error: No such container: gpu4pyscf
C02G8BSLMD6R:manylinux bytedance$ docker rmi b533a80e387e
Untagged: cupy/cupy:v11.3.0
Untagged: cupy/cupy@sha256:4efb9c9d73ef51230dd2d12785da1553e21e76dc67efff83ca20d7a74a988424
Deleted: sha256:b533a80e387ee58847ca38b562cc7ac69a19bb4fb023e79f5e76244de7700b7d
C02G8BSLMD6R:manylinux bytedance$ docker rmi 1e56485ef4b4
Error response from daemon: conflict: unable to delete 1e56485ef4b4 (must be forced) - image is being used by stopped container df3ecb8542ee
C02G8BSLMD6R:manylinux bytedance$ 
C02G8BSLMD6R:manylinux bytedance$ 
C02G8BSLMD6R:manylinux bytedance$ docker rmi 5681298be998
Deleted: sha256:5681298be9986c3391dbd058c4766ee18c2d955fa028535b4c08a426e7795981
C02G8BSLMD6R:manylinux bytedance$ 
C02G8BSLMD6R:manylinux bytedance$ 
C02G8BSLMD6R:manylinux bytedance$ 
C02G8BSLMD6R:manylinux bytedance$ docker build .
[+] Building 573.7s (13/16)                                                                                                                                                                                                                                   
 => [internal] load build definition from Dockerfile                                                                                                                                                                                                     0.0s
 => => transferring dockerfile: 37B                                                                                                                                                                                                                      0.0s
 => [internal] load .dockerignore                                                                                                                                                                                                                        0.0s
 => => transferring context: 2B                                                                                                                                                                                                                          0.0s
 => [internal] load metadata for quay.io/pypa/manylinux2014_x86_64:2023-09-24-36b93e4                                                                                                                                                                    0.3s
 => [ 1/12] FROM quay.io/pypa/manylinux2014_x86_64:2023-09-24-36b93e4@sha256:07145d872015f593222ac57874e0ef14f3f4bde5b391ff4d1be8979c28170d7e                                                                                                            0.0s
 => [internal] load build context                                                                                                                                                                                                                        0.0s
 => => transferring context: 37B                                                                                                                                                                                                                         0.0s
 => CACHED [ 2/12] RUN yum install -y wget curl perl util-linux xz bzip2 git patch which perl zlib-devel                                                                                                                                                 0.0s
 => CACHED [ 3/12] RUN yum install -y yum-utils centos-release-scl                                                                                                                                                                                       0.0s
 => CACHED [ 4/12] RUN rm -rf /opt/python/cp26-cp26m /opt/_internal/cpython-2.6.9-ucs2                                                                                                                                                                   0.0s
 => CACHED [ 5/12] RUN rm -rf /opt/python/cp26-cp26mu /opt/_internal/cpython-2.6.9-ucs4                                                                                                                                                                  0.0s
 => CACHED [ 6/12] RUN rm -rf /opt/python/cp33-cp33m /opt/_internal/cpython-3.3.6                                                                                                                                                                        0.0s
 => CACHED [ 7/12] RUN rm -rf /opt/python/cp34-cp34m /opt/_internal/cpython-3.4.6                                                                                                                                                                        0.0s
 => CACHED [ 8/12] ADD install_cuda.sh install_cuda.sh                                                                                                                                                                                                   0.0s
 => ERROR [ 9/12] RUN bash ./install_cuda.sh 11.8 && rm install_cuda.sh                                                                                                                                                                                573.3s
------                                                                                                                                                                                                                                                        
 > [ 9/12] RUN bash ./install_cuda.sh 11.8 && rm install_cuda.sh:                                                                                                                                                                                             
#12 0.302 + test 1 -gt 0                                                                                                                                                                                                                                      
#12 0.302 + case "$1" in                                                                                                                                                                                                                                      
#12 0.302 + install_118                                                                                                                                                                                                                                       
#12 0.302 + echo 'Installing CUDA 11.8 and cuDNN 8.7 and NCCL 2.15 and cutensor 1.6.1.5'                                                                                                                                                                      
#12 0.302 Installing CUDA 11.8 and cuDNN 8.7 and NCCL 2.15 and cutensor 1.6.1.5
#12 0.302 + rm -rf /usr/local/cuda-11.8 /usr/local/cuda
#12 0.303 + wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
#12 457.8 + chmod +x cuda_11.8.0_520.61.05_linux.run
#12 457.8 + ./cuda_11.8.0_520.61.05_linux.run --toolkit --silent
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Libraries_11.8-components/CUDA_Runtime_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Tools_11.8-components/CUDA_Command_Line_Tools_11.8-components/ can't be opened
#12 572.1 /var/log/nvidia/.uninstallManifests/CUDA_Toolkit_11.8-components/CUDA_Tools_11.8-components/CUDA_Command_Line_Tools_11.8-components/ can't be opened
#12 572.1 terminate called after throwing an instance of 'boost::filesystem::filesystem_error'
#12 572.1   what():  boost::filesystem::copy_file: No such file or directory: "./builds/cuda_cupti/extras/CUPTI/samples/pc_sampling_start_stop/pc_sampling_start_stop.cu", "/usr/local/cuda-11.8/extras/CUPTI/samples/pc_sampling_start_stop/pc_sampling_start_stop.cu"
#12 572.1 ./cuda_11.8.0_520.61.05_linux.run: line 524:    42 Aborted                 ./cuda-installer --toolkit --silent
------
executor failed running [/bin/sh -c bash ./install_cuda.sh ${BASE_CUDA_VERSION} && rm install_cuda.sh]: exit code: 134
C02G8BSLMD6R:manylinux bytedance$ 
C02G8BSLMD6R:manylinux bytedance$ 
C02G8BSLMD6R:manylinux bytedance$ 
C02G8BSLMD6R:manylinux bytedance$ emacs Dockerfile
C02G8BSLMD6R:manylinux bytedance$ emacs install_cuda.sh
install_cuda.sh   install_cuda.sh~  
C02G8BSLMD6R:manylinux bytedance$ emacs install_cuda.sh

File Edit Options Buffers Tools Sh-Script Help                                                                                                                                                                                                                
        export GENCODE=$OVERRIDE_GENCODE
    fi

    # all CUDA libs except CuDNN and CuBLAS                                                                                                                                                                                                                   
    ls $CUDA_LIB_DIR/ | grep "\.a" | grep -v "culibos" | grep -v "cudart" | grep -v "cudnn" | grep -v "cublas" | grep -v "metis"  \
      | xargs -I {} bash -c \
                "echo {} && $NVPRUNE $GENCODE $CUDA_LIB_DIR/{} -o $CUDA_LIB_DIR/{}"

    # prune CuDNN and CuBLAS                                                                                                                                                                                                                                  
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublas_static.a -o $CUDA_LIB_DIR/libcublas_static.a
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublasLt_static.a -o $CUDA_LIB_DIR/libcublasLt_static.a

    #####################################################################################                                                                                                                                                                     
    # CUDA 12.1 prune visual tools                                                                                                                                                                                                                            
    #####################################################################################                                                                                                                                                                     
    export CUDA_BASE="/usr/local/cuda-12.1/"
    rm -rf $CUDA_BASE/libnvvp $CUDA_BASE/nsightee_plugins $CUDA_BASE/nsight-compute-2023.1.0 $CUDA_BASE/nsight-systems-2023.1.2/
}

# idiomatic parameter and option handling in sh                                                                                                                                                                                                               
while test $# -gt 0
do
    case "$1" in
    11.8) install_118; prune_118
                ;;
    12.1) install_121; prune_121
                ;;
        *) echo "bad argument $1"; exit 1
           ;;
    esac
    shift
done
