// Copyright MediaZ AS. All Rights Reserved.

#include "module.Curve/ZD.Curve.h"

#include <glm/glm.hpp>
#include <nlohmann/json.hpp>
#include <boost/math/interpolators/pchip.hpp>
#include <boost/math/interpolators/barycentric_rational.hpp>
#include <boost/math/interpolators/makima.hpp>
#include <functional>

using namespace std;
using namespace std::placeholders;
using namespace boost::math::interpolators;
using json = nlohmann::json;

static double clamp(const double& value, const double& min, const double& max)
{
    double rv = value;
    if (rv > max)
        rv = max;
    else if (rv < min)
        rv = min;
    return rv;
}

namespace ZD::Curve
{

    void CalibrationCurve::SetCPCount(int size)
    {
        X.resize(size);
        Y.resize(size);
    }

    int CalibrationCurve::GetCPCount() const
    {
        return (int)X.size();
    }

    double CalibrationCurve::GetX(int index) const
    {
        return X[index];
    }
    
    double CalibrationCurve::GetY(int index) const
    {
        return Y[index];
    }

    Point CalibrationCurve::GetPoint(int index) const
    {
        return Point{ X[index],  Y[index] };
    }

    double* CalibrationCurve::GetXArray()
    {
        return &X[0];
    }

    double* CalibrationCurve::GetYArray()
    {
        return &Y[0];
    }

    void CalibrationCurve::SetX(int index, double value)
    {
        X[index] = value;
        _interpolator = nullptr; // will be rebuild next time it is required
    }

    void CalibrationCurve::SetY(int index, double value)
    {
        Y[index] = value;
        _interpolator = nullptr; // will be rebuild next time it is required
    }

    void CalibrationCurve::SetInterpolationMethod(InterpolationMethod method)
    {
        _interpolationMethod = method;
        _interpolator = nullptr;
    }

    InterpolationMethod CalibrationCurve::GetInterpolationMethod()
    {
        return _interpolationMethod;
    }

    void CalibrationCurve::GetInterpolatedCurve(vector<double>& x, vector<double>& y, int length)
    {
        x.clear();
        y.clear();
        for (int i = 0; i < length; i++)
        {
            double tmp = clamp(i / (double)length, 0, 1);
            x.push_back(tmp);
            y.push_back(GetInterpolatedValue(tmp));
        }
    }

    int CalibrationCurve::InsertCP(double x)
    {
        if (x < 0 || x > 1.0)
            return -1;

        double f = GetInterpolatedValue(x);
        int index = InsertCP(x, f);
        rebuild();
        return index;
    }

    int CalibrationCurve::InsertCP(double x, double value)
    {
        for (int i = 0; i < X.size(); i++)
        {
            if (abs(X[i] - x) < 0.05)
                return -1;
        }

        _interpolator = nullptr;

        if (X.size() == 0)
        {
            X.push_back(x);
            Y.push_back(value);
            return (int)X.size() - 1;
        }
        else if (X[0] > x)
        {
            X.insert(begin(X), x);
            Y.insert(begin(Y), value);
            return 0;
        }
        else if (X.back() < x)
        {
            X.push_back(x);
            Y.push_back(value);
            return (int)X.size() - 1;
        }
        else
        {
            for (int i = 0; i < X.size() - 1; i++)
            {
                if (X[i] < x && X[i + 1] > x)
                {
                    X.insert(begin(X) + i + 1, x);
                    Y.insert(begin(Y) + i + 1, value);
                    return i + 1;
                }
            }
            return -1; // what?
        }
    }

    void CalibrationCurve::RemoveCP(int index)
    {
        if ((index < 0) || (index >= X.size()))
            return;

        X.erase(X.begin() + index);
        Y.erase(Y.begin() + index);
        rebuild();
    }

    void CalibrationCurve::ModifyCP(int zoomIndex, double zoomValue)
    {
        if (zoomValue < 0)
            zoomValue = 0;
        else if (zoomValue >= 1.0)
            zoomValue = 1.0;
        if (zoomIndex != 0 && zoomIndex != X.size() - 1) // if not the first and not the last
        {
            if (zoomValue <= X[zoomIndex - 1])
                zoomValue = X[zoomIndex - 1] + 0.01f;

            else if (zoomValue >= X[zoomIndex + 1])
                zoomValue = X[zoomIndex + 1] - 0.01f;
        }
        X[zoomIndex] = zoomValue;
    }

    bool CalibrationCurve::LoadCurve(const std::vector<double>& vx, const std::vector<double>& vy)
    {
        if (vx.size() != vy.size())
            return false;
        X.clear();
        Y.clear();
        for (int i = 0; i < (int)vx.size(); i++)
        {
            X.push_back(vx[i]);
            Y.push_back(vy[i]);
        }
        if (X.size() >= 2)
        {
            if (X.front() < 0)
                X.front() = 0.0;
            if (X.back() > 1)
                X.back() = 1.0;
        }
        return true;
    }

    void CalibrationCurve::ExtrapolateLastItem()
    {
        if (X.size() > 2)
        {
            int last = (int)X.size() - 1;

            glm::vec2 v1e(X[last - 2], Y[last - 2]);
            glm::vec2 v2e(X[last - 1], Y[last - 1]);

            double m = (v2e.y - v1e.y) / (v2e.x - v1e.x);
            double val = m * (1.0 - v2e.x) + v2e.y;

            Y[last] = val;
            X[last] = 1.0;
        }
    }

    void CalibrationCurve::rebuild()
    {
        _interpolator = nullptr; // will be rebuild next time it is required
    }

    double CalibrationCurve::get_extrapolated_value(double x)
    {
        if (X.size() == 0)
            return 0.0;
        if (X.size() == 1)
            return Y[0];
        if (X.size() >= 2)
        {
            if (x < 0)
            {
                glm::vec2 v1e(0.1, GetInterpolatedValue(0.1));
                glm::vec2 v2e(0.0, GetInterpolatedValue(0.0));

                double m = (v2e.y - v1e.y) / (v2e.x - v1e.x);
                double val = m * x + v2e.y;
                return val;
            }
            if (x > 1.0)
            {
                glm::vec2 v1e(0.9, GetInterpolatedValue(0.9));
                glm::vec2 v2e(1.0, GetInterpolatedValue(1.0));

                double m = (v2e.y - v1e.y) / (v2e.x - v1e.x);
                double val = m * (1.0 - x) + v2e.y;
                return val;
            }
        }
        return 0; // we will never hit here, but let's put this to supres warning C4715
    }

    double CalibrationCurve::GetInterpolatedValue(double x)
    {
        x = x * _multiplier;
        if (x < 0 || x > 1.0)
            return get_extrapolated_value(x);

        if (X.size() == 0)
            return 0.0;

        if (X.size() == 1)
            return Y[0];

        if (X.size() == 2)
        {
            double slope = (Y[1] - Y[0]) / (X[1] - X[0]);
            double r = (slope * (x - X[1])) + Y[1];
            return (double)r;
        }
        if (X.size() == 3)
        {
            vector<double> xt = X;
            vector<double> yt = Y;
            auto spline = barycentric_rational(std::move(xt), std::move(yt), 2);
            return spline(x);
        }
        if (_interpolator == nullptr)
        {
            vector<double> x = X;
            vector<double> y = Y;
            switch (_interpolationMethod)
            {
            case InterpolationMethod::PCHIP:
            {
                auto spline = new pchip(move(x), move(y));
                _interpolator = bind(*spline, _1); }
            break;
            case InterpolationMethod::Barycentric_2:
            {
                auto spline = new barycentric_rational(move(x), move(y), 2);
                _interpolator = bind(*spline, _1); }
            break;
            case InterpolationMethod::Barycentric_3:
            {
                auto spline = new barycentric_rational(move(x), move(y));
                _interpolator = bind(*spline, _1);
            }
            break;
            case InterpolationMethod::Makima:
            {
                auto spline = new makima(move(x), move(y));
                _interpolator = bind(*spline, _1);
            }
            break;
            }
        }
        return _interpolator(x);
    }

    void CalibrationCurve::reset()
    {
        X.clear();
        Y.clear();
        _interpolator = nullptr;
    }
}