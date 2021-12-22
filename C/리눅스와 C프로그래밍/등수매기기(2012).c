#include<stdio.h>
#include <algorithm>
using namespace std;
int main(){
	int j,n,i,number,temp;
	long long sum = 0;
	scanf("%d",&n);
	int rank[n] = {0};
	for(i=0;i<n;i++){
		scanf("%d",&rank[i]);
	}
	sort(rank,rank+n);
	for(i=0;i<n;i++){
		sum+= abs(rank[i] - (i+1));
	}
	printf("%lld",sum);
}
