import streamlit as st
import requests
import joblib
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

# 🔹 Load ML model
model = joblib.load("safety_model.pkl")

# 🔹 API Keys
GRAPH_HOPPER_API_KEY = "341a134e-aa35-49b8-9d1d-a9081df68769"
WEATHER_API_KEY = "d328c036d12036b542eb5ddeedecbc67"

# 🔹 Geolocator
geolocator = Nominatim(user_agent="route_safety_app", timeout=10)

# 🔹 Session state
if "prediction" not in st.session_state:
    st.session_state["prediction"] = None
if "source_coords" not in st.session_state:
    st.session_state["source_coords"] = None
if "dest_coords" not in st.session_state:
    st.session_state["dest_coords"] = None

# ------------------- Functions -------------------

def get_coordinates(location_name):
    try:
        location = geolocator.geocode(location_name)
        return [location.latitude, location.longitude] if location else None
    except:
        return None

def get_weather(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "condition": data['weather'][0]['main'],
            "temperature": data['main']['temp'],
            "visibility": data.get('visibility', 'N/A')
        }
    return {"condition": "Unavailable", "temperature": "N/A", "visibility": "N/A"}

def get_alternative_routes(source_coords, dest_coords):
    url = (
        f"https://graphhopper.com/api/1/route"
        f"?point={source_coords[0]},{source_coords[1]}"
        f"&point={dest_coords[0]},{dest_coords[1]}"
        f"&vehicle=car&locale=en&points_encoded=false"
        f"&algorithm=alternative_route&alternative_route.max_paths=3"
        f"&key={GRAPH_HOPPER_API_KEY}"
    )

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        routes = []

        for path in data.get("paths", []):
            coords = path["points"]["coordinates"]
            latlng_coords = [[lat, lon] for lon, lat in coords]  # flip to [lat, lon]

            routes.append({
                "travel_time": path["time"] / 60000,
                "distance": path["distance"] / 1000,
                "points": latlng_coords
            })

        return routes
    else:
        st.error("Error fetching routes from GraphHopper.")
        st.text(response.text)  # Useful for debugging
    return []


print("Model expects these features:", model.feature_names_in_)


def predict_safety(source_coords, dest_coords):
    src_weather = get_weather(*source_coords)
    dest_weather = get_weather(*dest_coords)
    alternative_routes = get_alternative_routes(source_coords, dest_coords)

    safest_route = None
    safest_score = float("inf")

    for route in alternative_routes:
        # 🔹 Ensure feature order matches the model's training data
        input_df = pd.DataFrame([{
            "Total Accidents - 2015": 0,
            "Killed - 2015": 0,
            "Injured - 2015": 0,
            "Severity - 2015": 0
        }], columns=model.feature_names_in_)  # 🔹 Force exact column order
        
        prediction = model.predict(input_df)[0]

        if prediction == 0 and route["travel_time"] < safest_score:
            safest_route = route
            safest_score = route["travel_time"]

    return safest_route, alternative_routes, src_weather, dest_weather



# ------------------- Streamlit UI -------------------

st.set_page_config(page_title="Smart Route Safety AI", layout="centered")
st.title("🚦 AI-Powered Route Safety Predictor")
st.markdown("Enter your source and destination to automatically check route safety!")

source = st.text_input("Enter Source Location:")
destination = st.text_input("Enter Destination Location:")

if st.button("Check Route Safety"):
    st.session_state["source_coords"] = get_coordinates(source)
    st.session_state["dest_coords"] = get_coordinates(destination)

    if st.session_state["source_coords"] and st.session_state["dest_coords"]:
        st.session_state["prediction"] = predict_safety(
            st.session_state["source_coords"],
            st.session_state["dest_coords"]
        )
    else:
        st.error("❌ Could not find coordinates for one or both locations.")

# 🔹 Display Results
if st.session_state["prediction"]:
    safest_route, alternative_routes, src_weather, dest_weather = st.session_state["prediction"]

    st.subheader("🌤 Weather Conditions")
    st.write(f"📍 **Source:** {src_weather['condition']} | {src_weather['temperature']}°C")
    st.write(f"🏁 **Destination:** {dest_weather['condition']} | {dest_weather['temperature']}°C")

    st.subheader("🛣 Safest Suggested Route")
    if safest_route:
        travel_minutes = safest_route['travel_time']
        travel_hours = travel_minutes / 60
        travel_days = travel_hours / 24

        formatted_time = f"{int(travel_days)} days, {int(travel_hours % 24)} hours, {int(travel_minutes % 60)} minutes"

        st.success(f"✅ Travel Time: {formatted_time}\n📍 Distance: {safest_route['distance']:.2f} km")


    else:
        st.warning("⚠️ No completely safe routes found. Showing alternatives instead.")

    st.markdown("### 🚀 Alternative Routes")
    for i, route in enumerate(alternative_routes):
        st.write(f"🔹 **Route {i+1}:** {route['travel_time']:.2f} mins, {route['distance']:.2f} km")

    # 🔹 Map
    m = folium.Map(location=st.session_state["source_coords"], zoom_start=13)
    folium.Marker(st.session_state["source_coords"], popup="Source", icon=folium.Icon(color="blue")).add_to(m)
    folium.Marker(st.session_state["dest_coords"], popup="Destination", icon=folium.Icon(color="red")).add_to(m)

    for i, route in enumerate(alternative_routes):
        if route["points"]:
            folium.PolyLine(locations=route["points"], color="blue", weight=3, tooltip=f"Route {i+1}").add_to(m)
        else:
            st.warning(f"⚠️ Route {i+1} has no valid points to display.")

    if safest_route and safest_route["points"]:
        folium.PolyLine(locations=safest_route["points"], color="green", weight=5, tooltip="Safest Route").add_to(m)

    st_folium(m, width=700, height=500)
