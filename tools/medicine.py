import requests
from crewai.tools import tool

@tool("medicine_tool")
def medicine_tool(medicine_name: str) -> str:
    """Fetches medicine availability from backend API"""
    try:
        response = requests.get(
            f'https://172.16.63.49:8000/medicines/name/{medicine_name}/',
            verify=False  
        )

        if response.status_code != 200:
            return f"Sorry! Couldn't fetch availability for {medicine_name}."

        data = response.json()

        medicine_info = data.get("medicine", {})
        instances = data.get("instances", [])

        if not instances:
            return f"⚠️ {medicine_name} is currently not available in nearby pharmacies."

        result = f"🩺 Availability for **{medicine_name.title()}**:\n\n"

        for store in instances:
            pharmacy = store.get("pharmacy", {})
            result += (
                f"🏪 {pharmacy.get('pharmacy_name', 'Unknown Pharmacy')}\n"
                f"📍 {pharmacy.get('location', 'Unknown Location')}\n"
                f"📞 {pharmacy.get('contact_no', 'N/A')}\n"
                "──────────────\n"
            )

        return result.strip()

    except Exception as e:
        return f"❌ Error contacting backend: {str(e)}"