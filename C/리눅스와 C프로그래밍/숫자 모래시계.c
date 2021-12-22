#include<stdio.h>
int main(){
    int num,i,j;
    scanf("%d",&num);
    for(i=1;i<=num;i++){
            printf("\n");
        for(j=1;j<i;j++){
        printf(" ");
        }
        for(j=i;j<=num;j++){
            printf("%d",j);
        }
        for(j=num-1;j>=i;j--){
            printf("%d",j);
        }

    }
    for(i=1;i<num;i++){
            printf("\n");
        for(j=num-i;j>1;j--){
            printf(" ");
        }
        for(j=num-i;j<=num;j++){
            printf("%d",j);
        }
        for(j=num-1;j>=num-i;j--){
            printf("%d",j);
        }

    }
}

