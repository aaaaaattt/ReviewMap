import streamlit as st
import openai
import faiss
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
import folium
from streamlit_folium import st_folium
from dotenv import load_dotenv
import os


# .env 파일 로드
load_dotenv()

# OpenAI 및 Google Maps API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")  # .env 파일에서 OpenAI API 키 가져오기
google_maps_api_key = os.getenv("API_KEY")  # .env 파일에서 Google Maps API 키 가져오기

# FAISS 및 데이터 로드
faiss_index_path = "faiss_index.bin"  # 저장된 FAISS 파일 경로
csv_data_path = "places_reviews_with_embeddings(final_2).csv"  # 메타데이터 CSV 파일 경로

index = faiss.read_index(faiss_index_path)  # FAISS 인덱스 로드
metadata = pd.read_csv(csv_data_path)  # 장소 정보 로드


def get_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    # 응답에서 임베딩 데이터 추출
    return response['data'][0]['embedding']



# Google Maps에서 위치 정보 가져오기
def get_location(name, address):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={name},+{address}&key={google_maps_api_key}"
    response = requests.get(url).json()
    if response['results']:
        location = response['results'][0]['geometry']['location']
        return location['lat'], location['lng']
    return None, None

# Streamlit 애플리케이션
st.title("장소 추천 및 지도 표시 서비스")
user_input = st.text_input("검색어를 입력하세요", placeholder="예: 24시 운동복 대여 가능한 헬스장")

if user_input:
    # 사용자 입력 임베딩
    st.write("입력 텍스트 임베딩 생성 중...")
    query_embedding = np.array(get_embedding(user_input)).astype('float32').reshape(1, -1)
    
    # FAISS에서 유사도 계산
    st.write("유사도 계산 중...")
    distances, indices = index.search(query_embedding, k=5)  # 상위 5개 결과 반환
    
    # 상위 결과 추출
    results = metadata.iloc[indices[0]].copy()
    results['similarity'] = 1 - distances[0] / 2  # 코사인 유사도 계산 (1 - L2 거리 / 2)

    # 추천된 장소 및 리뷰 표시
    st.write("추천된 장소 및 리뷰:")
    st.dataframe(results[['name', 'address', 'review_text', 'similarity']])

    # Google Maps Embed (동적 지도) 생성
    st.write("**Google Maps 동적 지도**")
    
    # JSON 형태로 변환
    locations = []
    for _, row in results.iterrows():
        lat, lng = get_location(row['name'], row['address'])
        if lat and lng:
            locations.append({
                "name": row['name'],
                "address": row['address'],
                "review_text": row['review_text'],
                "similarity": row['similarity'],
                "latitude": lat,
                "longitude": lng
            })
    
        # HTML 및 JavaScript 생성
        html_code = f"""
        <!DOCTYPE html>
        <html>
          <head>
            <title>추천된 장소 지도</title>
            <script async defer src="https://maps.googleapis.com/maps/api/js?key={google_maps_api_key}&callback=initMap"></script>
            <script>
              function initMap() {{
                const map = new google.maps.Map(document.getElementById('map'), {{
                  zoom: 12,
                  center: {{ lat: 37.5665, lng: 126.9780 }}  // 서울 중심 좌표
                }});

                const bounds = new google.maps.LatLngBounds();
                const locations = {locations};

                locations.forEach((location) => {{
                  if (location.latitude && location.longitude) {{
                    const marker = new google.maps.Marker({{
                      position: {{ lat: location.latitude, lng: location.longitude }},
                      map: map,
                      title: location.name,
                      icon: {{
                        path: google.maps.SymbolPath.CIRCLE,
                        scale: 20,
                        fillColor: "rgb(255, 0, 0)",
                        fillOpacity: 0.9,
                        strokeWeight: 1,
                        strokeColor: "#000"
                      }}
                    }});

                    const infoWindow = new google.maps.InfoWindow({{
                      content: `
                        <div style="max-width: 200px;">
                          <h3>${{location.name}}</h3>
                          <p>주소: ${{location.address}}</p>
                          <p>리뷰: ${{location.review_text}}</p>
                          <p>유사도: ${{(location.similarity * 100).toFixed(2)}}%</p>
                        </div>`
                    }});

                    marker.addListener('click', () => {{
                      infoWindow.open(map, marker);
                    }});

                    bounds.extend(marker.position);
                  }}
                }});

                map.fitBounds(bounds);
              }}
            </script>
          </head>
          <body>
            <div id="map" style="width: 100%; height: 500px;"></div>
          </body>
        </html>
        """

    # Streamlit에서 HTML 출력
    st.components.v1.html(html_code, height=600)



