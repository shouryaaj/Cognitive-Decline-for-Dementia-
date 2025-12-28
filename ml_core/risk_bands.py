def chi_to_risk(chi):
    if chi >= 70:
        return "Green (Good)"
    elif chi >= 40:
        return "Yellow (Monitor)"
    else:
        return "Red (High Risk)"
