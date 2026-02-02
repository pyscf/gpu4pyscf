# Show friendly names recorded in SPIR-V and search for the string
for f in *.spv; do
    echo $f
    /opt/aurora/25.190.0/oneapi/compiler/2025.2/bin/compiler/llvm-spirv -to-text -o - "$f" | grep 'lda'
done
