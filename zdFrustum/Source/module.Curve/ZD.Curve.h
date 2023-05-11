/*
 * Copyright MediaZ AS. All Rights Reserved.
 */

#pragma once

#include <vector>
#include <string>
#include <functional>
#include <glm/glm.hpp>
namespace ZD::Curve
{
    enum class InterpolationMethod
    {
        PCHIP,
        Barycentric_2,
        Barycentric_3,
        Makima
    };
    
    struct Point
    {
        double X;
        double Y;
    };

    class CalibrationCurve
    {
    public:

        void SetCPCount(int size);
        int GetCPCount()  const;

        Point GetPoint(int cpIndex) const;

        double GetX(int cpIndex) const;
        double GetY(int cpIndex)  const;

        void SetX(int cpIndex, double value);
        void SetY(int cpIndex, double value);

        double* GetXArray();
        double* GetYArray();
       
        void SetInterpolationMethod(InterpolationMethod method);
        InterpolationMethod GetInterpolationMethod();

        double GetMultiplier();
        void SetMultiplier(double value);

        bool LoadCurve(const std::vector<double>& vx, const std::vector<double>& vy);

    public:      
        int		InsertCP(double x); 
        int		InsertCP(double x, double value);
        void	RemoveCP(int index);
        void	ModifyCP(int index, double newValue);
        double	GetInterpolatedValue(double x);
        void	rebuild();
        void	reset();

    public: // tools
        void GetInterpolatedCurve(std::vector<double>& x, std::vector<double>& y, int sampleCount = 100);
        void ExtrapolateLastItem();
    private:
        std::vector<double>	X;
        std::vector<double>	Y;
        std::function<double(double)> _interpolator;
        void* _previousInterpolatorObject;
        double get_extrapolated_value(double x);
        friend class GenericCalibration;
        InterpolationMethod _interpolationMethod = InterpolationMethod::PCHIP;
        double _multiplier = 1.0f;

    };

    class GenericCalibration
    {
    public:
        std::string			Name;

        double				ImageHeight;
        double				ImageWidth;

        double				FocusRingStartOffset;
        double				FocalLengthOffset;

        CalibrationCurve	FocalLengthCurve;
        double				FocalLengthScaler = 1.0;

        CalibrationCurve	NodalPointCurve;

        CalibrationCurve	FocusDistanceCurve;

        CalibrationCurve	FocusCurve;
        double				FocusCurveFocus = -1.0;

        CalibrationCurve	K1Curve1;
        double				K1Curve1Focus;

        CalibrationCurve	K1Curve2;
        double				K1Curve2Focus;

        CalibrationCurve	K2Curve1;
        double				K2Curve1Focus;

        CalibrationCurve	K2Curve2;
        double				K2Curve2Focus;

    public:
        GenericCalibration();
        void Initialize(double minFocalLength, double maxFocalLength);
        
        double GetFoV(double zoom, double focus);
        glm::dvec2 GetK1K2(double zoom, double focus);

        void reevaluate();
    
        std::string Serialize();
        void Deserialize(std::string s);

    };

    std::vector<double> GetInterpolatedFoVCurve(GenericCalibration& clb, double focus);
    void GetInterpolatedK1Curve(GenericCalibration& clb, double focus, std::vector<double>& x, std::vector<double>& y);
    void GetInterpolatedK2Curve(GenericCalibration& clb, double focus, std::vector<double>& x, std::vector<double>& y);
};
