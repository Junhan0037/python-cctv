#!/usr/bin/env python3 
#pandas 기초 익히기 (서울시 구별 CCTV 현황 분석)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#list 생성
s = pd.Series([1,3,5,np.nan,6,8])
print(s)

dates = pd.date_range('20130101', periods=6)
print(dates)

df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=['A','B','C','D'])
print(df)

#행,열,값,정보 출력
print(df.index)
print(df.columns)
print(df.values)
print(df.info())

#통계적 개요(count, mean, max, min, 표준편차 등) 출력
print(df.describe())

#by로 지정된 칼럼을 기준으로 정렬, ascending옵션으로 내림차순, 오름차순 정렬
print(df.sort_values(by='B', ascending=False))

#특정 칼럼 값들 출력
print(df['A'])
print(df[0:3])

#특정 행들을 출력
print(df['20130102':'20130104'])

#특정 행,열 출력
print(df.loc[dates[0]])
print(df.loc[:,['A','B']])
print(df.loc['20130102':'20130104',['A','B']])
print(df.loc['20130102',['A','B']])
print(df.loc[dates[0],'A'])

#특정 행,열 출력2
print(df.iloc[3])
print(df.iloc[3:5, 0:2])
print(df.iloc[[1,2,4],[0,2]])
print(df.iloc[1:3,:])
print(df.iloc[:,1:3])

#조건에 맞는 행,열 출력
print(df[df.A > 0])
print(df[df > 0])

#df2=df 명령어는 데이터의 위치만 복사되기 때문에 데이터의 내용까지 복사하려면 copy() 사용
df2 = df.copy()

#새로운 열 추가
df2['E'] = ['one', 'two', 'three', 'four', 'five', 'six']
print(df2)

#isin조건 안에 있는 변수들이 해당 열에 존재하는지 여부
print(df2['E'].isin(['two','four']))

#isin조건에 해당하는 열만 출력
print(df2[df2['E'].isin(['two','four'])])

#누적합 통계
print(df.apply(np.cumsum))

#최대값과 최소값의 차이
print(df.apply(lambda x : x.max() - x.min()))

df1 = pd.DataFrame({'A':['A0', 'A1', 'A2', 'A3'],
                    'B':['B0', 'B1', 'B2', 'B3'],
                    'C':['C0', 'C1', 'C2', 'C3'],
                    'D':['D0', 'D1', 'D2', 'D3']},
                    index=[0,1,2,3])
df2 = pd.DataFrame({'A':['A4', 'A5', 'A6', 'A7'],
                    'B':['B4', 'B5', 'B6', 'B7'],
                    'C':['C4', 'C5', 'C6', 'C7'],
                    'D':['D4', 'D5', 'D6', 'D7']},
                    index=[4,5,6,7])
df3 = pd.DataFrame({'A':['A8', 'A9', 'A10', 'A11'],
                    'B':['B8', 'B9', 'B10', 'B11'],
                    'C':['C8', 'C9', 'C10', 'C11'],
                    'D':['D8', 'D9', 'D10', 'D11']},
                    index=[8,9,10,11])

print(df1)
print(df2)
print(df3)

#열방향으로 합치기
result = pd.concat([df1, df2, df3])
print(result)

#다중index설정을 위한 keys옵션
result = pd.concat([df1, df2, df3], keys=['x', 'y', 'z'])
print(result)

#level을 이용하여 출력
print(result.index)
print(result.index.get_level_values(0))
print(result.index.get_level_values(1))

df4 = pd.DataFrame({'B':['B2', 'B3', 'B6', 'B7'],
                    'D':['D2', 'D3', 'D6', 'D7'],
                    'F':['F2', 'F3', 'F6', 'F7']},
                    index=[2,3,6,7])

#합치는데 index를 기준으로 맞지 않는곳은 Nan으로 표시
result = pd.concat([df1, df4], axis=1)
print(result)

#공통되지 않는 index데이터 삭제(Nan행열 삭제)
result = pd.concat([df1, df4], axis=1, join='inner')
print(result)

#df1의 인덱스를 기준으로 합치기
result = pd.concat([df1, df4], axis=1, join_axes=[df1.index])
print(result)

#두 데이터의 index를 무시하고 합친 후, 다시 index 부여(ignore_index=True)
result = pd.concat([df1, df4], ignore_index=True)
print(result)

left = pd.DataFrame({'key':['K0', 'K4', 'K2', 'K3'],
                     'A':['A0', 'A1', 'A2', 'A3'],
                     'B':['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key':['K0', 'K1', 'K2', 'K3'],
                     'C':['C0', 'C1', 'C2', 'C3'],
                     'D':['D0', 'D1', 'D2', 'D3']})
#두 데이터의 공통된 칼럼인 값으로 합치기
print(pd.merge(left, right, on='key'))
print(pd.merge(left, right, how='left', on='key'))
print(pd.merge(left, right, how='right', on='key'))
print(pd.merge(left, right, how='outer', on='key'))
print(pd.merge(left, right, how='inner', on='key'))

#시각화 도구
plt.figure
plt.plot([1,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1,0])
plt.show()

#0~12까지 0.01간격으로 데이터 생성
t = np.arange(0,12,0.01)
#sin값
y = np.sin(t)

#그래프 출력
plt.figure(figsize=(10,6))
plt.plot(t, y)
plt.show()

#그래프 출력
plt.figure(figsize=(10,6))
plt.plot(t, y)
#격자무늬
plt.grid()
#x축 이름
plt.xlabel('time')
#y축 이름
plt.ylabel('Amplitude')
#title 이름
plt.title('Example of sinewave')
plt.show()

#두줄의 그래프 출력
plt.figure(figsize=(10,6))
plt.plot(t, np.sin(t))
plt.plot(t, np.cos(t))
plt.grid()
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('Example of sinewave')
plt.show()

#두줄의 그래프 출력 (라벨 추가)
plt.figure(figsize=(10,6))
plt.plot(t, np.sin(t), label='sin')
plt.plot(t, np.cos(t), label='cos')
plt.grid()
#라벨 표시
plt.legend()
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('Example of sinewave')
plt.show()

#두줄의 그래프 출력 (라벨 추가, 선 굵기 및 색깔 추가)
plt.figure(figsize=(10,6))
#선 굵기(순서 중요!)
plt.plot(t, np.sin(t), lw=3, label='sin')
#선 색깔(순서 중요!)
plt.plot(t, np.cos(t), 'r', label='cos')
plt.grid()
plt.legend()
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('Example of sinewave')
plt.show()

t = [0,1,2,3,4,5,6]
y = [1,4,5,8,9,5,3]

plt.figure(figsize=(10,6))
#선 색깔, 선 스타일, 데이터가 존재하는 곳에 마킹, 마킹 색깔, 마킹 사이즈
plt.plot(t, y, color='green', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=12)
#x축, y축의 최소값과 최대값을 지정
plt.xlim([-0.5, 6.5])
plt.ylim([0.5, 9.5])
plt.show()

t = np.array([0,1,2,3,4,5,6,7,8,9])
y = np.array([9,8,7,9,8,3,2,4,3,4])

plt.figure(figsize=(10,6))
#점으로만 표시, 마킹 모양
plt.scatter(t, y, marker='>')
plt.show()

#x축의 값인 t에 따라 색상을 바꾸기
colormap = t
plt.figure(figsize=(10,6))
#s옵션 마커의 크기
plt.scatter(t, y, s=50, c=colormap, marker='>')
plt.colorbar()
plt.show()

#랜덤변수 함수 생성, loc옵션 평균값, scale옵션 표준편차 지정
s1 = np.random.normal(loc=0, scale=1, size=1000)
s2 = np.random.normal(loc=5, scale=0.5, size=1000)
s3 = np.random.normal(loc=10, scale=2, size=1000)

plt.figure(figsize=(10,6))
plt.plot(s1, label='s1')
plt.plot(s2, label='s2')
plt.plot(s3, label='s3')
plt.legend()
plt.show()

#boxplot 그래프 생성
plt.figure(figsize=(10,6))
plt.boxplot((s1, s2, s3))
plt.grid()
plt.show()







