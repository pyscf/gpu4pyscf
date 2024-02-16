import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def parse_results():
    with open('scan-final.xyz', 'r') as file:
        # Read all lines in the file
        lines = file.readlines()

    # List to hold lines containing 'Dihedral'
    dihedral_lines = []

    # Loop through each line in the file
    for line in lines:
        # If 'Dihedral' is found in the line, add it to the list
        if re.search('Dihedral', line, re.IGNORECASE):
            dihedral_lines.append(line.rstrip())

    records = []
    for dl in dihedral_lines:
        cycle_txt, dihedral_txt, iteration_energy_txt = dl.split(';')
        iteration_txt, energy_txt = iteration_energy_txt.split('Energy')
        dihedral_degree = float(dihedral_txt.strip().split(' ')[-1])
        iteration_number = int(iteration_txt.strip().split(' ')[-1])
        energy_hartree = float(energy_txt.split(' ')[-1])
        cycle_number = int(cycle_txt.strip().split('/')[0].split(' ')[-1])
        records.append(
            [cycle_number, dihedral_degree, iteration_number, energy_hartree])

    return records


def plot_scan(records: list):
    degrees = [x[1] for x in records]
    iteration_numbers = [x[2] for x in records]
    hartree = [x[3] for x in records]

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Dihedral (degree)')
    ax1.set_ylabel('Energy (kcal/mol)', color=color)
    ax1.plot(degrees,
             627.5096080305927 * (hartree - np.min(hartree)),
             'x-',
             color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Geometric Optimization Iterations',
                   color=color)  # we already handled the x-label with ax1
    ax2.plot(degrees, iteration_numbers, 'o-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.title('Energy and Geometric Optimization Iterations')
    fig.tight_layout()  # to adjust subplots to fit into the figure area.

    plt.savefig('scan_result.png', dpi=150)


def main():
    records = parse_results()
    plot_scan(records)
    for r in records:
        print(r)


main()
