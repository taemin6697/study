#include <stdio.h>
#include <stdlib.h>
int main()
{
    int n,i,j,k=0;
    int day,pay;
    int array1[100] = {};
    int array2[100] = {};
    int sum[100] = {0};
    int big = 0;
    int y;
    scanf("%d",&n);
    for(i=0;i<n;i++){
        scanf("%d %d",&day,&pay);
        array1[i] = day;
        array2[i] = pay;
    }
    for(j=0;j<n;j++){
        for(i=j;i<n;i+=array1[i]){
            if(i<n){
            sum[k] = sum[k] + array2[i];
            y = i;
            }
            else if(i<n){
                continue;
            }
        }
        if(i>n){
            sum[k] = sum[k] - array2[y];
        }
        k++;
    }
    for(i=0;i<100;i++){
        if(big<sum[i]){
            big = sum[i];
        }
    }
    printf("%d",big);
}
