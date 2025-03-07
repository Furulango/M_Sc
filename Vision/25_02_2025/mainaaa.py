#Programa para generar numeros aleatorios

import random

def main():
    print("Numeros aleatorios")
    print("Presiona enter para generar un numero aleatorio")
    print("Presiona 'q' para salir")
    while True:
        key = input()
        if key == 'q':
            break
        print(random.randint(1, 100))

if __name__ == "__main__":
    main()

        