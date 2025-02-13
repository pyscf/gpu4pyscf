from pyscf.gto.mole import cart2sph

def gen_l(l):
    c2s = cart2sph(l)
    m, n = c2s.shape
    for j in range(n):
        s = []
        for i in range(m):
            if abs(c2s[i,j]) > 1e-16:
                s.append(f'{c2s[i,j]}*gcart[{i}]')
        print(f'gsph[{j}] = ' + ' + '.join(s) + ';')


gen_l(3)
gen_l(4)
gen_l(5)
gen_l(6)
gen_l(7)
gen_l(8)
