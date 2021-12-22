/** 다음 단계로 넘어갈 때 A는 B로, B는 AB로 바뀌니까
	현 단계의 B갯수만큼 다음 단계의 A가 생성되고
	현 단계의 A갯수 + B갯수 만큼 다음 단계의 B가 생성됨*/
#include<stdio.h>
#define MAX 45
int main(){
	int num;
	int numA[MAX],numB[MAX];
	scanf("%d",&num);
	numA[0] = 1,numB[0] = 0;
	for(int i=0;i<num;i++){
		int temp = numA[i]; //1 0
		numA[i+1] = numB[i]; // 0 1
		numB[i+1] = temp+numB[i];//1 1
	}
	printf("%d %d",numA[num],numB[num]);
}
