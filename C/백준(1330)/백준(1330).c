#define _CRT_SECURE_NO_WARININGS
#include <stdio.h>
int main() {

    char x;
    printf("문자를 입력하세요:");
    scanf("%c", &x);



    if (x == 'c')
        printf("Circle");
    else if (x == 't')
        printf("Triangle");
    else if (x == 'r')
        printf("Rectangle");
    else
        printf("Unknown");


    return 0;
}