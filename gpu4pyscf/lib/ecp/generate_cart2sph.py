from pyscf.gto.mole import cart2sph

def gen_cart2sph(l):
    c2s = cart2sph(l)
    m, n = c2s.shape
    for j in range(n):
        s = []
        for i in range(m):
            if abs(c2s[i,j]) > 1e-16:
                s.append(f'{c2s[i,j]}*gcart[{i}]')
        print(f'gsph[{j}] = ' + ' + '.join(s) + ';')

gen_cart2sph(3)
gen_cart2sph(4)
gen_cart2sph(5)
gen_cart2sph(6)
gen_cart2sph(7)
gen_cart2sph(8)
gen_cart2sph(9)
gen_cart2sph(10)

def gen_sph2cart(l):
    c2s = cart2sph(l)
    m, n = c2s.shape
    for j in range(m):
        s = []
        for i in range(n):
            if abs(c2s[j,i]) > 1e-16:
                s.append(f'{c2s[j,i]}*gsph[{i}]')
        print(f'gcart[{j}] = ' + ' + '.join(s) + ';')

gen_sph2cart(2)
gen_sph2cart(3)
gen_sph2cart(4)
gen_sph2cart(5)
gen_sph2cart(6)
gen_sph2cart(7)
gen_sph2cart(8)
gen_sph2cart(9)
gen_sph2cart(10)
