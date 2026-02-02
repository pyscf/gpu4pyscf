# Reverse SPIR-V to bitcode and list defined functions
for f in *.spv; do
  bc="${f%.spv}.bc"
  if llvm-spirv -r -o "$bc" "$f" >/dev/null 2>&1; then
    /opt/aurora/25.190.0/oneapi/compiler/2025.2/bin/compiler/llvm-nm --defined-only --demangle "$bc" | awk -vF="$f" '/ [Tt] /{print F, $3}'
  else
    # Fallback: look for friendly names in OpName
    llvm-spirv -to-text -o - "$f" 2>/dev/null | awk -vF="$f" '/OpName/ {print F, $0}'
  fi
done > /tmp/devsyms.txt

# Many Intel *.bin are ELF containers; try nm; else fall back to strings
for f in *.bin; do
  if /opt/aurora/25.190.0/oneapi/compiler/2025.2/bin/compiler/llvm-nm --defined-only "$f" >/dev/null 2>&1; then
    /opt/aurora/25.190.0/oneapi/compiler/2025.2/bin/compiler/llvm-nm --defined-only --demangle "$f" | awk -vF="$f" '/ [Tt] /{print F, $3}'
  else
    strings "$f" | grep -E 'func_vxc_unpol|gga_x_fd_lb94|_vxc_unpol' | awk -vF="$f" '{print F, $0}'
  fi
done >> /tmp/devsyms.txt

# Do we have duplicate *function* names across device images?
cut -d' ' -f2 /tmp/devsyms.txt | sort | uniq -d > /tmp/dev_dups.txt
