from pyscf.gto.mole import cart2sph

def gen_l(l):
    n = 0
    for lx in reversed(range(l+1)):
        for ly in reversed(range(l+1-lx)):
            lz = l - lx - ly
            s = ['rx'] * lx
            s += ['ry'] * ly
            s += ['rz'] * lz
            print(f'double g{n} = ' + '*'.join(s) + ';')
            n += 1

    c2s = cart2sph(l)
    m, n = c2s.shape
    for j in range(n):
        s = []
        for i in range(m):
            if abs(c2s[i,j]) > 1e-16:
                s.append(f'{c2s[i,j]}*g{i}')
        print(f'double c{j} = ' + ' + '.join(s) + ';')

    for i in range(m):
        s = []
        for j in range(n):
            if abs(c2s[i,j]) > 1e-16:
                s.append(f'{c2s[i,j]}*c{j}')
        print(f'omega[{i}] = ' + ' + '.join(s) + ';')

gen_l(4)
gen_l(5)
gen_l(6)
gen_l(7)
gen_l(8)
