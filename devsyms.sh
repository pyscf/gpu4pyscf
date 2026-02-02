# Optionally narrow the search to specific names to avoid noise:
# export PATTERN='func_vxc_unpol|func0_gga_x_fd_lb94|func1_gga_x_fd_lb94'
PATTERN="${PATTERN:-}"

NM=/opt/aurora/25.190.0/oneapi/compiler/2025.2/bin/compiler/llvm-nm
SPVREV=/opt/aurora/25.190.0/oneapi/compiler/2025.2/bin/compiler/llvm-spirv

out=output_devsyms.txt

emit_nm() { # $1=file $2=tag-for-origin
    local f="$1" tag="$2"
    if "$NM" --defined-only --demangle "$f" >/dev/null 2>&1; then
        if [[ -n "$PATTERN" ]]; then
            "$NM" --defined-only --demangle "$f" \
                | awk -vF="$tag" -vP="$PATTERN" '$2 ~ /^[Tt]$/ && $3 ~ P {print F, $3}'
        else
            "$NM" --defined-only --demangle "$f" \
                | awk -vF="$tag" '$2 ~ /^[Tt]$/ {print F, $3}'
        fi
    fi
}

# 1) Handle SPIR-V: reverse to .bc then run nm
for f in *.spv; do
    [[ -e "$f" ]] || continue
    bc="${f%.spv}.bc"
    if "$SPVREV" -r -o "$bc" "$f" >/dev/null 2>&1; then
        emit_nm "$bc" "$f" >> "$out"
    else
        # Fallback: textual SPIR-V to catch OpName (if supported)
        if "$SPVREV" -to-text -o "${f%.spv}.spvasm" "$f" >/dev/null 2>&1; then
            if [[ -n "$PATTERN" ]]; then
                grep -E "OpName %[^ ]+ \"($PATTERN)\"" "${f%.spv}.spvasm" \
                    | awk -vF="$f" '{print F, $NF}' >> "$out"
            else
                grep -E 'OpName %[^ ]+ "' "${f%.spv}.spvasm" \
                    | awk -vF="$f" '{print F, $NF}' >> "$out"
            fi
        fi
    fi
done

# 2) Handle .bin (often ZEBin/ELF): try nm; if not, fall back to strings
for f in *.bin; do
    [[ -e "$f" ]] || continue
    if "$NM" --defined-only "$f" >/dev/null 2>&1; then
        emit_nm "$f" "$f" >> "$out"
    else
        # Last resort: best-effort scan of readable names
        if [[ -n "$PATTERN" ]]; then
            strings "$f" | grep -E "$PATTERN" | awk -vF="$f" '{print F, $0}' >> "$out"
        else
            strings "$f" | grep -E 'func_|gga_' | awk -vF="$f" '{print F, $0}' >> "$out"
        fi
    fi
done

# # 3) Summarize duplicates
# cut -d' ' -f2 "$out" | sort | uniq -d > /tmp/dev_dups.txt

# echo "Device symbols  -> $out"
# echo "Duplicate names  -> /tmp/dev_dups.txt"
# [[ -s /tmp/dev_dups.txt ]] && echo "DUPLICATES FOUND" || echo "No duplicates found"
# SH
# chmod +x devsym_scan.sh
# ./devsym_scan.sh

