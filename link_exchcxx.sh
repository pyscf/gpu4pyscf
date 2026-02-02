# Make sure the deps dir exists
mkdir -p /home/abagusetty/gpu4pyscf-testing/gpu4pyscf/gpu4pyscf/lib/deps/lib

# Point libxc.so -> libgdft.so
ln -sf /home/abagusetty/gpu4pyscf-testing/gpu4pyscf/gpu4pyscf/lib/libgdft.so \
       /home/abagusetty/gpu4pyscf-testing/gpu4pyscf/gpu4pyscf/lib/deps/lib/libxc.so
