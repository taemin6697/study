#include<stdio.h>
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
int main(){
    int n;
    int number;
    scanf("%d",&n);
    int temp = 0;
    int c = 2;
    int big = 0;
    int num_list[100000]={0,};
    for(int i=0;i<n;i++){
        scanf("%d",&number);
        num_list[i] = number;
    }
    quickSort(num_list, 0, n - 1);
    for(int i=0;i<n;i++){
       if(big<num_list[i]+c){
        big = num_list[i]+c;
       }
       c++;

    }
    printf("%d",big);

}
