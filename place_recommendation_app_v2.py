import streamlit as st
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
import faiss
import time
import os
from openai import OpenAI

# 최상단에 배치
st.set_page_config(
    page_title="장소 추천 및 지도 표시 서비스",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# 스타일 추가
st.markdown("""
    <style>
        .css-1d391kg {
            width: 320px;
        }
        .css-1v0mbdj.e115fcil1 {
            display: block;
        }
        .sidebar .sidebar-content {
            width: 320px;
        }
    </style>
""", unsafe_allow_html=True)

# .env 파일 로드
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")

# FAISS 및 데이터 로드
faiss_index_path = "./faiss_index.bin"
csv_data_path = "./reviews_embeddings.csv"

index = faiss.read_index(faiss_index_path)
metadata = pd.read_csv(csv_data_path)

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def get_location(name, address, max_retries=3):
    for attempt in range(max_retries):
        try:
            url = f"https://maps.googleapis.com/maps/api/geocode/json?address={name},+{address}&key={google_maps_api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data['status'] == 'OK' and data['results']:
                location = data['results'][0]['geometry']['location']
                return location['lat'], location['lng']
            elif data['status'] == 'ZERO_RESULTS':
                st.warning(f"위치를 찾을 수 없습니다: {name}, {address}")
            else:
                st.error(f"Google Maps API 오류: {data['status']}")
            
            return None, None
        
        except requests.exceptions.RequestException as e:
            st.error(f"네트워크 오류 발생 (시도 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                st.error("위치 정보를 가져오는 데 실패했습니다.")
                return None, None

# Streamlit 애플리케이션
st.title("장소 추천 및 지도 표시 서비스")

 # 기존 사이드바 코드를 다음과 같이 수정
with st.sidebar:
    st.header("검색 설정")
    min_similarity = st.slider("최소 유사도 점수", 0.0, 1.0, 0.5, 0.01)
    num_results = st.slider("표시할 결과 개수", 1, 20, 5)

user_input = st.text_input("검색어를 입력하세요", placeholder="찾는 장소를 입력하세요.")

if user_input:
    # 사용자 입력 임베딩
    with st.spinner("입력 텍스트 임베딩 생성 중..."):
        query_embedding = np.array(get_embedding(user_input)).astype('float32').reshape(1, -1)
    
    # FAISS에서 유사도 계산
    with st.spinner("유사도 계산 중..."):
        distances, indices = index.search(query_embedding, k=num_results * 2)  # 필터링을 위해 더 많은 결과 검색
        
        # 상위 결과 추출 및 유사도 필터링
        results = metadata.iloc[indices[0]].copy()
        results['similarity'] = 1 - distances[0] / 2
        results = results[results['similarity'] >= min_similarity]
        results = results.head(num_results)

    if len(results) > 0:
        # 추천된 장소 및 리뷰 표시
        st.write("추천된 장소 및 리뷰:")
        st.dataframe(results[['name', 'address', 'review_text', 'similarity']])

        # Google Maps 동적 지도
        st.write("**Google Maps 동적 지도**")
        
        locations = []
        for _, row in results.iterrows():
            lat, lng = get_location(row['name'], row['address'])
            if lat and lng:
                locations.append({
                    "name": row['name'],
                    "address": row['address'],
                    "review_text": row['review_text'],
                    "similarity": float(row['similarity']),
                    "latitude": lat,
                    "longitude": lng
                })
        
        if locations:
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
                      center: {{ lat: 37.5665, lng: 126.9780 }}
                    }});

                    const bounds = new google.maps.LatLngBounds();
                    const locations = {locations};

                    locations.forEach((location) => {{
                      if (location.latitude && location.longitude) {{
                        const markerScale = 10 + (location.similarity * 20);
                        const redValue = Math.floor(255 * location.similarity);
                        const greenValue = Math.floor(255 * (1-location.similarity));
                        
                        const marker = new google.maps.Marker({{
                          position: {{ lat: location.latitude, lng: location.longitude }},
                          map: map,
                          title: location.name,
                          icon: {{
                            path: google.maps.SymbolPath.CIRCLE,
                            scale: markerScale,
                            fillColor: 'rgb(' + redValue + ',' + greenValue + ',0)',
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

            st.components.v1.html(html_code, height=600)
        else:
            st.warning("선택된 장소들의 위치 정보를 가져올 수 없습니다.")
    else:
        st.warning("설정된 유사도 기준을 만족하는 결과가 없습니다. 최소 유사도 점수를 낮추어 보세요.")