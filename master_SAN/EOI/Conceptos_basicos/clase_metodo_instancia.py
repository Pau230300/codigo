class my_clase:
    nombre='hola'
    value=8
    def funcion_1(self, value):
        self.value = value
        print(value)

    
    def funcion_2(self, num):
        num = self.value
        return num
    
    def funcion_3(self):
        return 3
    

# crear clase
class Dog:
    nombre = 'simba'
    edad = 1
    raza = 'golden retriever'
    energia = 15
    
    def sit(self, name_dog):
        if name_dog == self.nombre:
            print('El perro se ha sentado')
            self.energia -= 2
        else:
            print('El perro no te hace caso')
            self.energia -= 1
