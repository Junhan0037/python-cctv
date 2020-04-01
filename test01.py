#!/usr/bin/env python3 
#서울시 구별 CCTV 현황 분석
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#csv파일 읽기
CCTV_Seoul = pd.read_csv('CCTV_in_Seoul.csv', encoding='utf-8')

#5행까지 출력
print(CCTV_Seoul.head())

#열 출력
print(CCTV_Seoul.columns)
print(CCTV_Seoul.columns[0])

#열 이름 변경, inplace는 바뀐 이름을 변수에 적용
CCTV_Seoul.rename(columns={CCTV_Seoul.columns[0] : '구별'}, inplace=True)
print(CCTV_Seoul.head())

#excel파일 읽기
pop_Seoul = pd.read_excel('population_in_Seoul.xls', encoding='utf-8')
print(pop_Seoul.head())

#세번째 줄부터 읽기, 특정열 추출
pop_Seoul = pd.read_excel('population_in_Seoul.xls', header=2, parse_cols='B, D, G, J, N', encoding='utf-8')
print(pop_Seoul)

#열 이름 변경
pop_Seoul.rename(columns={pop_Seoul.columns[0] : '구별',
                          pop_Seoul.columns[1] : '인구수',
                          pop_Seoul.columns[2] : '한국인',
                          pop_Seoul.columns[3] : '외국인',
                          pop_Seoul.columns[4] : '고령자'}, inplace=True)
print(pop_Seoul.head())

#CCTV 소계기준 오름차순, 내림차순
print(CCTV_Seoul.sort_values(by='소계', ascending=True).head(5))
print(CCTV_Seoul.sort_values(by='소계', ascending=False).head(5))

#새로운 컬럼 추가, 최근 3년간 CCTV수를 덯고 2013년 이전 CCTV수로 나눠서 최근 3년간 CCTV증가율 계산
CCTV_Seoul['최근증가율'] = (CCTV_Seoul['2016년'] + CCTV_Seoul['2015년'] + CCTV_Seoul['2014년']) / CCTV_Seoul['2013년도 이전'] * 100
print(CCTV_Seoul.sort_values(by='최근증가율', ascending=False).head(5))

#첫번째 행 제거
print(pop_Seoul.drop([0], inplace=True))
print(pop_Seoul.head())

#unique를 이용하여 중복된 데이터 확인
print(pop_Seoul['구별'].unique())

#null값 존재여부
print(pop_Seoul['구별'].isnull())

pop_Seoul['외국인비율'] = pop_Seoul['외국인'] / pop_Seoul['인구수'] * 100
pop_Seoul['고령자비율'] = pop_Seoul['고령자'] / pop_Seoul['인구수'] * 100
print(pop_Seoul.head())

#특정열 기준 정
print(pop_Seoul.sort_values(by='인구수', ascending=False).head(5))
print(pop_Seoul.sort_values(by='외국인', ascending=False).head(5))

#'구별'컬럼을 기준으로 두 데이터 합치기
data_result = pd.merge(CCTV_Seoul, pop_Seoul, on='구별')
print(data_result.head())

#필요없는 컬럼 삭제
del data_result['2013년도 이전']
del data_result['2014년']
del data_result['2015년']
del data_result['2016년']
print(data_result.head())

#'구별'컬럼을 index로 설정
data_result.set_index('구별', inplace=True)
print(data_result.head())

#상관관계 분석
print(np.corrcoef(data_result['고령자비율'],data_result['소계']))
print(np.corrcoef(data_result['외국인비율'],data_result['소계']))
print(np.corrcoef(data_result['인구수'],data_result['소계']))

print(data_result.sort_values(by='소계', ascending=False).head(5))

#matplotlib 폰트 변경 (한글지원 X 때문)
import platform
from matplotlib import font_manager, rc
plt.rcParams['axes.unicode_minus'] = False
if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    path = "c:\Windows\Fonts\malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system... sorry~~~~')

print(data_result.head())
#CCTV개수 그래프 출력, kind옵션 수평바, grid옵션 격자무늬, figsize옵션 그래프 크기
data_result['소계'].plot(kind='barh', grid=True, figsize=(10,10))
plt.show()

#위의 그래프를 정렬하여 출력
data_result['소계'].sort_values().plot(kind='barh', grid=True, figsize=(10,10))
plt.show()

#인구대비 CCTV비율 그래프
data_result['CCTV비율'] = data_result['소계'] / data_result['인구수'] * 100
data_result['CCTV비율'].sort_values().plot(kind='barh', grid=True, figsize=(10,10))
plt.show()

#점 그래프
plt.figure(figsize=(6,6))
plt.scatter(data_result['인구수'], data_result['소계'], s=50) #s옵션 마킹의 크기
plt.xlabel('인구수')
plt.ylabel('CCTV')
plt.grid()
plt.show()

#1차함수(직선) 생성
fp1 = np.polyfit(data_result['인구수'], data_result['소계'], 1)
#y축 데이터 생성
f1 = np.poly1d(fp1)
#x축 데이터 생성
fx = np.linspace(100000, 700000, 100)

plt.figure(figsize=(10,10))
plt.scatter(data_result['인구수'], data_result['소계'], s=50)
plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g') #1차함수(직선) 생성, ls옵션 선모양, lw옵션 선굵기
plt.xlabel('인구수')
plt.ylabel('CCTV')
plt.grid()
plt.show()

#오차범위 생성 (1차함수에서 멀리 벗어난 순서)
data_result['오차'] = np.abs(data_result['소계'] - f1(data_result['인구수']))

df_sort = data_result.sort_values(by='오차', ascending=False)
print(df_sort.head())

plt.figure(figsize=(14,10))
plt.scatter(data_result['인구수'], data_result['소계'], c=data_result['오차'], s=50)
plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
for n in range(10):
    #오차범위가 큰 지역들 이름 붙이기
    plt.text(df_sort['인구수'][n]*1.02, df_sort['소계'][n]*0.98, df_sort.index[n], fontsize=15)
plt.xlabel('인구수')
plt.ylabel('인구당비율')
plt.colorbar()
plt.grid()
plt.show()

#결과
print('===== 결과 =====')
print('1. 강남구, 양천구, 서초구, 용산구는 서울시 전체 지역의 일반적인 경향보다 CCTV가 많이 설치된 지역')
print('2. 강북구, 도봉구, 광진구, 중랑구, 강서구, 송파구는 서울시 전체 지역의 일반적인 경향보다 CCTV가 적게 설치된 지역')

data_result.to_csv('CCTV_result.csv', sep=',', encoding='utf-8')
