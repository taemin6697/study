#include <stdio.h>
#include <stdlib.h>

int main()
{
    int need_temp;
    int need_water;
    int min_su = 0;
    scanf("%d %d",&need_temp,&need_water);//10 10
    if(need_temp>need_water){
        min_su = need_water;
    }
    else{
        min_su = need_temp;
    }
    min_su = min_su / 10;
    printf("%d",need_temp+need_water+min_su);

}
