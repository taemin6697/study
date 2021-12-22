#include<stdio.h>
#define MAX 31
int main(){
	int hang,k;
	scanf("%d %d",&hang,&k);
	int su[MAX][MAX];
	for(int i=0;i<31;i++){
		su[i][0] = 1;
	}
	for(int i=0;i<MAX;i++){
		for(int j=1;j<=i;j++){
			su[i][j] = su[i-1][j-1] + su[i-1][j];//i-1 C j-1

		}
	}
	printf("%d",su[hang-1][k-1]);

}
