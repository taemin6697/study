#include <stdio.h>
#include <stdlib.h>

void quickSort(int* data, int pos_start, int pos_end)
{
    if (pos_start >= pos_end) { // 정렬 하고자 하는 집단의 원소가 1개인 경우
        return;
    }

    int key = pos_start; // 키는 첫번째 원소
    int i = pos_start + 1;
    int j = pos_end;
    int temp;

    while (i <= j) { // 엇갈릴 때까지 반복
        while (data[i] >= data[key]) { // 키 값보다 큰 값을 만날 때까지 오른쪽으로 이동
            i++;
        }
        while (data[j] <= data[key] && j > pos_start) { // 키 값보다 작은 값을 만날 때까지 왼쪽으로 이동
            j--;
        }
        if (i > j) { // 현재 엇갈린 상태면 키 값과 교체
            temp = data[j];
            data[j] = data[key];
            data[key] = temp;
        } else {
            temp = data[j];
            data[j] = data[i];
            data[i] = temp;
        }
    }
    quickSort(data, pos_start, j - 1);
    quickSort(data, j + 1, pos_end);
}

int main()
{
    int n,i;
    int number;
    int number_list[100002] = {0,};
    int sum = 0;
    int count = 0;
    int sum1 = 0;
    scanf("%d",&n);
    for(i=0;i<n;i++){
        scanf("%d",&number_list[i]);
    }
    quickSort(number_list, 0, n - 1);
    for(i=0;i<(n/3)*2;i+=2){ //6
        sum = sum + number_list[i];//0 //2//4
        sum = sum + number_list[i+1];//1//3//5
        count += 1;
    }
    if(n<3){
        sum1 += number_list[0];
        sum1 += number_list[1];
    }
    if(n==3){
        printf("%d\n",sum);
    }
    else if(n % 3==1){
        sum += number_list[i];
        printf("%d\n",sum);
    }
    else if(n%3==2){//5
        sum += number_list[i];
        sum += number_list[i+1];
        printf("%d\n",sum);
    }
    else{
        printf("%d\n",sum);
    }
    return 0;

}
