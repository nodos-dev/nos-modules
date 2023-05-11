// Copyright MediaZ AS. All Rights Reserved.

#include "module.Curve/ZD.Curve.h"

#include <boost/math/interpolators/pchip.hpp>
#include <boost/math/interpolators/barycentric_rational.hpp>
#include <utility>
#include <cmath>
#include <glm/glm.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace std;

namespace ZD::Curve
{
	static double lerp(double x0, double y0, double x1, double y1, double x2)
	{
		// p0 = x0, y0
		// p1 = x1, y1
		// given a new x value (as arg x2), find and return y
		// this can both interpolate and also extrapolate
		if (abs((x1 - x0) * (y1 - y0)) < 0.000001)
		{
			return (y0 + y1) / 2;
		}
		double y = y0 + (x2 - x0) / (x1 - x0) * (y1 - y0);
		return y;
	}

	void FillDefaultValues(GenericCalibration& clb, double minFocalLength, double maxFocalLength)
	{
		if (clb.FocalLengthCurve.GetCPCount() == 0)
		{
			clb.FocalLengthCurve.InsertCP(0.0, minFocalLength);
			clb.FocalLengthCurve.InsertCP(1.0, maxFocalLength);
		}

		if (clb.FocusDistanceCurve.GetCPCount() == 0)
		{
			clb.FocusDistanceCurve.InsertCP(0.0, 1);
			clb.FocusDistanceCurve.InsertCP(1.0, 2);
		}

		if (clb.FocusCurve.GetCPCount() == 0)
		{
			clb.FocusCurve.InsertCP(0, 1);
			clb.FocusCurve.InsertCP(1, 1);
		}

		if (clb.NodalPointCurve.GetCPCount() == 0)
		{
			clb.NodalPointCurve.InsertCP(0, 0);
			clb.NodalPointCurve.InsertCP(1, 0);
		}

		if (clb.K1Curve1.GetCPCount() == 0)
		{
			clb.K1Curve1.InsertCP(0, 0);
			clb.K1Curve1.InsertCP(1, 0);
			clb.K1Curve1Focus = 0.0;
		}

		if (clb.K1Curve2.GetCPCount() == 0)
		{
			clb.K1Curve2.InsertCP(0, 0);
			clb.K1Curve2.InsertCP(1, 0);
			clb.K1Curve2Focus = 0.0;
		}

		if (clb.K2Curve1.GetCPCount() == 0)
		{
			clb.K2Curve1.InsertCP(0, 0);
			clb.K2Curve1.InsertCP(1, 0);
			clb.K2Curve1Focus = 0.0;
		}

		if (clb.K2Curve2.GetCPCount() == 0)
		{
			clb.K2Curve2.InsertCP(0, 0);
			clb.K2Curve2.InsertCP(1, 0);
			clb.K2Curve2Focus = 0.0;
		}
	}

	void serialize_curve(json& j, const string& curveName, const CalibrationCurve& curve)
	{
		j[curveName.c_str()]["X"] = json::array();
		j[curveName.c_str()]["Y"] = json::array();
		for (int i = 0; i < curve.GetCPCount(); i++)
		{
			j[curveName.c_str()]["X"].push_back(curve.GetX(i));
			j[curveName.c_str()]["Y"].push_back(curve.GetY(i));
		}
	}

	void deserialize_curve(json& j, const string& curveName, CalibrationCurve& curve)
	{
		if (j.contains(curveName.c_str()))
		{
			vector<double> x;
			vector<double> y;
			j[curveName.c_str()]["X"].get_to(x);
			j[curveName.c_str()]["Y"].get_to(y);
			curve.LoadCurve(x, y);
		}
	}

	void GetInterpolatedK1Curve(GenericCalibration& clb, double focus, vector<double>& x, vector<double>& y)
	{
		x.clear();
		y.clear();
		for (int i = 0; i <= 100; i++)
		{
			x.push_back(i / 100.0);
			y.push_back(clb.GetK1K2(i / 100.0, focus).x);
		}
	}

	void GetInterpolatedK2Curve(GenericCalibration& clb, double focus, vector<double>& x, vector<double>& y)
	{
		x.clear();
		y.clear();
		for (int i = 0; i <= 100; i++)
		{
			x.push_back(i / 100.0);
			y.push_back(clb.GetK1K2(i / 100.0, focus).y);
		}
	}

	vector<double> GetInterpolatedFoVCurve(GenericCalibration& clb, double focus)
	{
		vector<double> curve;
		for (int i = 0; i <= 100; i++)
			curve.push_back(clb.GetFoV(i / 100.0, focus));
		return curve;
	}

	GenericCalibration::GenericCalibration()
	{
		// load some sane defaults
		ImageHeight = 5.4f; // standard for the 2/3" sensor
		ImageWidth = 9.6f;  // standard for the 2/3" sensor
		FocalLengthOffset = 0;

		Initialize(8.0, 90.0);
	}

	glm::dvec2 GenericCalibration::GetK1K2(double zoom, double focus)
	{
		double k1 = 0;
		{
			double k1_1 = K1Curve1.GetInterpolatedValue(zoom);
			if (K1Curve2Focus < 0.001)
				k1 = k1_1; // we aren't properly initialized.
			else
			{
				double k1_2 = K1Curve2.GetInterpolatedValue(zoom);
				k1 = lerp(K1Curve1Focus, k1_1, K1Curve2Focus, k1_2, focus);
			}
		}
		double k2 = 0;
		{
			double k2_1 = K2Curve1.GetInterpolatedValue(zoom);
			if (K2Curve2Focus < 0.001)
				k2 = k2_1; // we aren't properly initialized.
			else
			{
				double k2_2 = K2Curve2.GetInterpolatedValue(zoom);
				k2 = lerp(K2Curve1Focus, k2_1, K2Curve2Focus, k2_2, focus);
			}
		}
		return { k1, k2 };
	}

	double GenericCalibration::GetFoV(double zoom, double focus)
	{
		// get focal length at this zoom, at focus=infinity
		double focalLength = FocalLengthCurve.GetInterpolatedValue(zoom + FocalLengthOffset);
		// get FoV for focal length above
		double fov = (double)(2 * atan(((ImageHeight / 2.0) / focalLength)) * 180.0 / 3.14159265358979);
		// get FoV at this focus
		double fovMultiplier_FocusInf = 1.0;
		double fovMultiplier_FocusNow = FocusCurve.GetInterpolatedValue(zoom);

		// interpolate between f=inf (1.0) and f=this(focusMultiplier)
		//pt0
		auto actualFocusMultiplier = lerp(0, fovMultiplier_FocusInf, FocusCurveFocus, fovMultiplier_FocusNow, focus);
		return fov * actualFocusMultiplier;
	}

	void GenericCalibration::Initialize(double minFocalLength, double maxFocalLength)
	{
		Name = "";
		FocalLengthCurve.reset();
		FocusCurve.reset();
		NodalPointCurve.reset();
		FocusDistanceCurve.reset();

		K1Curve1.reset(); 
		K1Curve1Focus = 0.0;
		K1Curve2.reset();
		K1Curve2Focus = 0.0;

		K2Curve1.reset();
		K2Curve1Focus = 0.0;
		K2Curve2.reset();
		K2Curve2Focus = 0.0;

		FocusCurveFocus = -1;
		FillDefaultValues(*this, minFocalLength, maxFocalLength);
	}

	void GenericCalibration::reevaluate()
	{
		FocusCurve.rebuild();
		FocalLengthCurve.rebuild();
	}

	string GenericCalibration::Serialize()
	{
		json j;
		j["Name"] = Name;
		j["FocalDistanceOffset"] = FocalLengthOffset;
		j["FocusRingStartOffset"] = FocusRingStartOffset;
		j["ImageHeight"] = ImageHeight;
		j["ImageWidth"] = ImageWidth;

		serialize_curve(j, "FocalLengthCurve", FocalLengthCurve);
		serialize_curve(j, "NodalPointCurve", NodalPointCurve);
		serialize_curve(j, "FocusDistanceCurve", FocusDistanceCurve);
		serialize_curve(j, "FocusCurve", FocusCurve);

		serialize_curve(j, "K1Curve1", K1Curve1);
		serialize_curve(j, "K1Curve2", K1Curve2);
		serialize_curve(j, "K2Curve1", K2Curve1);
		serialize_curve(j, "K2Curve2", K2Curve2);

		j["FocusCurveFocus"] = FocusCurveFocus;
		j["K1Curve1Focus"] = K1Curve1Focus;
		j["K1Curve2Focus"] = K1Curve2Focus;
		j["K2Curve1Focus"] = K2Curve1Focus;
		j["K2Curve2Focus"] = K2Curve2Focus;

		return j.dump();
	}

	void GenericCalibration::Deserialize(string s)
	{
		json j = json::parse(s);
		Name = "";
		FocalLengthOffset = 0.0;
		FocusRingStartOffset = 0.0;
		ImageHeight = 5.4;
		ImageWidth = 9.6;
		K1Curve1Focus = 0.0;
		K1Curve2Focus = 0.0;
		K2Curve1Focus = 0.0;
		K2Curve2Focus = 0.0;

		FocalLengthCurve.reset();
		FocusCurve.reset(); FocusCurveFocus = 0.0;
		NodalPointCurve.reset();
		K1Curve1.reset(); K1Curve1Focus = 0.0;
		K1Curve2.reset(); K1Curve2Focus = 0.0;
		K2Curve1.reset(); K2Curve1Focus = 0.0;
		K2Curve2.reset(); K2Curve2Focus = 0.0;

		if (j.contains("Name"))
			Name = j["Name"];

		if (j.contains("FocalDistanceOffset"))
			FocalLengthOffset = j["FocalDistanceOffset"];

		if (j.contains("FocusRingStartOffset"))
			FocusRingStartOffset = j["FocusRingStartOffset"];

		if (j.contains("ImageHeight"))
			ImageHeight = j["ImageHeight"];

		if (j.contains("ImageWidth"))
			ImageWidth = j["ImageWidth"];

		deserialize_curve(j, "FocalLengthCurve", FocalLengthCurve);
		deserialize_curve(j, "FocusCurve", FocusCurve);
		deserialize_curve(j, "NodalPointCurve", NodalPointCurve);
		deserialize_curve(j, "FocusDistanceCurve", FocusDistanceCurve);

		deserialize_curve(j, "K1Curve1", K1Curve1);
		deserialize_curve(j, "K1Curve2", K1Curve2);
		deserialize_curve(j, "K2Curve1", K2Curve1);
		deserialize_curve(j, "K2Curve2", K2Curve2);

		if (FocusDistanceCurve.X.size() < 2)
		{
			FocusDistanceCurve.InsertCP(0, 1);
			FocusDistanceCurve.InsertCP(1, 1);
		}
		if (NodalPointCurve.X.size() < 2)
		{
			NodalPointCurve.InsertCP(0, 1);
			NodalPointCurve.InsertCP(1, 1);
		}

		if (K1Curve1.X.size() < 2)
		{
			K1Curve1.InsertCP(0, 0);
			K1Curve1.InsertCP(1, 0);
		}

		if (K1Curve2.X.size() < 2)
		{
			K1Curve2.InsertCP(0, 0);
			K1Curve2.InsertCP(1, 0);
		}

		if (K2Curve1.X.size() < 2)
		{
			K2Curve1.InsertCP(0, 0);
			K2Curve1.InsertCP(1, 0);
		}

		if (j.contains("FocusCurveFocus"))
			FocusCurveFocus = j["FocusCurveFocus"];

		if (j.contains("K1Curve1Focus"))
			K1Curve1Focus = j["K1Curve1Focus"];
		if (j.contains("K1Curve2Focus"))
			K1Curve2Focus = j["K1Curve2Focus"];
		if (j.contains("K2Curve1Focus"))
			K2Curve1Focus = j["K2Curve1Focus"];
		if (j.contains("K2Curve2Focus"))
			K2Curve2Focus = j["K2Curve2Focus"];
	}
}
