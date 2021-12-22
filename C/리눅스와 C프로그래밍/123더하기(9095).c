#include<stdio.h>
#include <algorithm>

using namespace std;

void sum_123(int N){
	int dp[12];
	dp[0] = 0,dp[1] = 1;
	dp[2] = 2,dp[3] = 4;
	for(int i=4;i<12;i++){
		dp[i] = dp[i-1] +dp[i-2] + dp[i-3];
	}
	printf("%d\n",dp[N]);
}

int main(){
	int N;
	int num,i;
	scanf("%d",&N);
	 for (i=0; i < N; i++)
    {
        scanf("%d", &num);
        sum_123(num);
    }
}
