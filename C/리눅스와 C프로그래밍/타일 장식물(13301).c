#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>

long long dp[81] = { 1, 1, };

long long fibo(int n)
{
    if (n == 0) return 1;
    if (n == 1) return 1;
    if (dp[n] != 0)return dp[n];

    for (int i = 2; i <= 80; i++)
    {
        dp[i] = dp[i - 1] + dp[i - 2];
    }

    return dp[n];
}

int main()
{
    int n;

    scanf("%d", &n);
    fibo(n);

    printf("%lld\n", (dp[n - 1] * 2) + (dp[n - 1] + dp[n - 2]) * 2);

    return 0;
}
