#include <digio.h>
#include <uart.h>

#include <xQuP01v0.h>

void main(void){
    
    // Configurar UART para Bluetooth (115200 bps)
    uart_setup(UART_B115200);
    
    while(1){
        // Enviar solo números y letras básicas
        uart_println("123");
        delay_ms(1000);
        
        uart_println("ABC");
        delay_ms(1000);
        
        uart_println("abc");
        delay_ms(1000);
        
        uart_println("Test");
        delay_ms(1000);
    }
}