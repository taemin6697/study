#include<stdio.h>
int main(){
	int n,num;
	int go;
	int fruit[10001];
	int i,j;
	scanf("%d %d",&n,&num);
	for(i=0;i<n;i++){
		scanf("%d",&go);
		fruit[i] = go;
	}
	for(i=0;i<n-1;i++){
		for(j=0;j<n-1;j++){
			if(fruit[j]>fruit[j+1]){
				int temp = fruit[j];
				fruit[j] = fruit[j+1];
				fruit[j+1] = temp;
			}
		}
	}
	for(i=0;i<n;i++){
		if(num>=fruit[i]){
			num++;
		}
		else{
			break;
		}
	}
	printf("%d",num);
}
