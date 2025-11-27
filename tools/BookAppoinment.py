from crewai.tools import tool
import requests

@tool("bookDoctor")
def medicine_tool(doctor_name: str,patien_id:int) -> str:
    """Fetches medicine availability from backend API"""
    try:
        response = requests.get(
            f'https://127.0.0.1:8000/medicines/search/?q={medicine_name}',
            verify=False  # skip SSL verification for local dev
        )

        if response.status_code != 200:
            return f"Sorry! Couldn't fetch availability for {medicine_name}."

        data = response.json()
        medicine_info = data.get("medicine_found", {})

        if not medicine_info:
            return f"⚠️ {medicine_name} is currently not found in the database."

        instances = medicine_info.get("instances", [])
        if not instances:
            return f"⚠️ {medicine_name} is currently not available in nearby pharmacies."

        result = f"🩺 Availability for **{medicine_info.get('medicine_name', medicine_name).title()}**:\n\n"

        for store in instances:
            pharmacy = store.get("pharmacy", {})
            result += (
                f"pharmacy_name :{pharmacy.get('pharmacy_name', 'Unknown Pharmacy')}\n"
                f"location :{pharmacy.get('location', 'Unknown Location')}\n"
                f"contact_no :{pharmacy.get('contact_no', 'N/A')}\n"
            )

        return result.strip()

    except Exception as e:
        return f"❌ Error contacting backend: {str(e)}"