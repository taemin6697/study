#include <stdio.h>
#include <stdlib.h>

int main()
{
   int num1,num2;
   int i,min,max,j;
   int k,y=1;
   scanf("%d %d",&num1,&num2);
   if(num2>num1){
    max = num2;
    min = num1;
   }
   else{
    max = num1;
    min = num2;
   }
    for(i=max;i>1;i--){
        if(num1%i==0 && num2%i==0){
            y = i;
            break;
        }


   }
   for(j=min;;j++){
    if(j%num1==0 && j%num2==0){
        k = j;
        break;
    }
   }
   printf("%d\n",y);
   printf("%d\n",k);


}
