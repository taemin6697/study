#include <iostream>

using namespace std;
bool average(int x[],int num,int &avg){
for(int i=0;i<num;i++){

    avg += x[i];
}
    avg /= num;
    if(avg<=0){
        return false;
    }
    else{

        return true;
    }


}


int main(void)

{

	int x[] = { 4, 8, 16, 32 };

	int avg = 0;



	if (average(x, 4, avg))

		cout << avg << endl;

	else

		cout << "매개변수 오류" << endl;



	if (average(x, -2, avg))

		cout << avg << endl;

	else

		cout << "매개변수 오류" << endl;

	return 0;

}
