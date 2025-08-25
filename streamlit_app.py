# -*- coding: utf-8 -*-
"""
Streamlit App for Visualizing Asia Sea Surface Temperature (SST)

- Data Source: NOAA OISST v2 High-Resolution (via OPeNDAP, PSL THREDDS)
- Key Libraries: Streamlit, Xarray, Matplotlib
- Cartopy: 선택적(있으면 지도 투영, 없으면 lat/lon 평면 폴백)
"""
import os
from datetime import date, timedelta

import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import TwoSlopeNorm

# ---- Cartopy (선택적) ----
HAS_CARTOPY = False
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

# --- 1) 기본 설정 & 폰트 ---
st.set_page_config(page_title="아시아 해수면 온도(SST) 대시보드", layout="centered")

def setup_korean_font():
    """
    fonts/Pretendard-Bold.ttf 가 있으면 등록하여 한글 깨짐 방지.
    없으면 시스템 기본 폰트를 시도.
    """
    font_path = os.path.join("fonts", "Pretendard-Bold.ttf")
    if os.path.exists(font_path):
        try:
            fm.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = "Pretendard"
        except Exception:
            plt.rcParams["font.sans-serif"] = ["Pretendard", "Malgun Gothic", "Apple SD Gothic Neo", "DejaVu Sans"]
            plt.rcParams["font.family"] = "sans-serif"
    else:
        st.warning("⚠️ `fonts/Pretendard-Bold.ttf`가 없어 기본 폰트를 사용합니다. (한글이 약간 달라 보일 수 있어요)")
        plt.rcParams["font.sans-serif"] = ["Malgun Gothic", "Apple SD Gothic Neo", "DejaVu Sans"]
        plt.rcParams["font.family"] = "sans-serif"

    plt.rcParams["axes.unicode_minus"] = False

# --- 2) 데이터 로딩 ---
@st.cache_data(show_spinner="해수면 온도 데이터 로딩 중...")
def load_sst_data(selected_date: date):
    """
    NOAA OISST v2 high-res(PSL THREDDS)에서 선택 날짜의 아시아 영역만 로드.
    - 항상 '연도별 파일 sst.day.mean.{YYYY}.nc'만 사용 (NRT 없음)
    - 아시아 영역(lat: -10~60, lon: 60~150)
    """
    year = selected_date.year
    data_url = f"https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.{year}.nc"
    st.info(f"데이터 소스: `{data_url}`")

    # pydap 엔진 먼저 시도 → 실패 시 기본 엔진
    engines_to_try = ["pydap", None]
    last_err = None

    for eng in engines_to_try:
        try:
            ds = xr.open_dataset(data_url, engine=eng)
            if "sst" not in ds:
                raise KeyError("변수 'sst'를 찾을 수 없습니다.")

            asia_sst = ds["sst"].sel(
                time=np.datetime64(selected_date),
                lat=slice(-10, 60),
                lon=slice(60, 150),
            ).squeeze()

            data = asia_sst.load()
            ds.close()
            return data
        except Exception as e:
            last_err = e
            continue

    st.error(
        "데이터 로딩 실패 😢\n\n"
        f"- 마지막 오류: {last_err}\n"
        "가능한 원인\n"
        "1) 인터넷/방화벽 문제\n"
        "2) NOAA 서버 지연/점검 중\n"
        "3) 선택 날짜 데이터가 아직 집계되지 않음(특히 최근 일자)\n"
        "→ 날짜를 1~3일 더 이전으로 바꿔보세요."
    )
    return None

# --- 3) 시각화 ---
def create_map_with_cartopy(sst_data: xr.DataArray, selected_date: date):
    fig = plt.figure(figsize=(10, 9))
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    # 배경/해안선
    ax.add_feature(cfeature.LAND, zorder=1, edgecolor='k', facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE, zorder=1, edgecolor='black')
    ax.set_extent([60, 150, -10, 60], crs=proj)

    # 그리드 라벨(버전 호환)
    try:
        gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
    except Exception:
        gl = ax.gridlines(draw_labels=True)
        try:
            gl.xlabels_top = False
            gl.ylabels_right = False
        except Exception:
            pass

    # 30°C 강조
    norm = TwoSlopeNorm(vmin=20, vcenter=30, vmax=34)
    cmap = "YlOrRd"

    mesh = sst_data.plot.pcolormesh(
        ax=ax,
        transform=proj,
        cmap=cmap,
        norm=norm,
        add_colorbar=False,
    )

    cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', pad=0.05, shrink=0.75)
    cbar.set_label("해수면 온도 (°C)")

    ax.set_title(f"해수면 온도: {selected_date.strftime('%Y년 %m월 %d일')}", fontsize=16, weight='bold')
    return fig

def create_simple_latlon_plot(sst_data: xr.DataArray, selected_date: date):
    """
    Cartopy가 없을 때 사용하는 평면(lat/lon) 플롯.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    norm = TwoSlopeNorm(vmin=20, vcenter=30, vmax=34)
    cmap = "YlOrRd"

    lon = sst_data["lon"].values
    lat = sst_data["lat"].values
    mesh = ax.pcolormesh(lon, lat, sst_data.values, cmap=cmap, norm=norm, shading="auto")

    cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', pad=0.02, shrink=0.9)
    cbar.set_label("해수면 온도 (°C)")

    ax.set_xlabel("경도")
    ax.set_ylabel("위도")
    ax.set_title(f"(Cartopy 미사용) 해수면 온도: {selected_date.strftime('%Y년 %m월 %d일')}", fontsize=14, weight='bold')
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig

# --- 4) Streamlit UI ---
def main():
    setup_korean_font()

    st.title("🌏 아시아 해수면 온도(SST) 대시보드")
    st.markdown("NOAA OISST v2 고해상도 데이터를 사용하여 아시아 전역의 **일별 해수면 온도**를 시각화합니다.")

    st.sidebar.header("🗓️ 날짜 선택")
    today = date.today()
    # 보통 2~3일 지연
    default_date = today - timedelta(days=3)
    min_date = date(1981, 9, 1)
    max_date = today - timedelta(days=2)

    selected_date = st.sidebar.date_input(
        "조회할 날짜",
        value=default_date,
        min_value=min_date,
        max_value=max_date,
        help=f"1981-09-01 ~ {max_date.strftime('%Y-%m-%d')}"
    )

    if selected_date:
        sst_data = load_sst_data(selected_date)

        if sst_data is not None and not sst_data.isnull().all():
            st.subheader(f"🗺️ {selected_date.strftime('%Y년 %m월 %d일')} 시각화")

            if HAS_CARTOPY:
                fig = create_map_with_cartopy(sst_data, selected_date)
            else:
                st.info("ℹ️ Cartopy가 없어 평면(lat/lon) 시각화로 표시합니다. (지도 경계선 없이 값 분포만 표시)")
                fig = create_simple_latlon_plot(sst_data, selected_date)

            st.pyplot(fig, clear_figure=True)

            with st.expander("📄 데이터 미리보기"):
                st.caption(
                    f"위도 {float(sst_data.lat.min()):.2f} ~ {float(sst_data.lat.max()):.2f}, "
                    f"경도 {float(sst_data.lon.min()):.2f} ~ {float(sst_data.lon.max()):.2f}"
                )
                st.write(sst_data)
        else:
            st.info("선택한 날짜의 데이터가 없거나 로딩에 실패했습니다. 다른 날짜를 선택해 주세요.")

if __name__ == "__main__":
    main()
