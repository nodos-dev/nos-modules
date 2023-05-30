float BlackWhitePoint(float Value, float BlackPoint, float WhitePoint)
{
    return (Value - BlackPoint) / (WhitePoint - BlackPoint);
}

float BlackWhitePointClamped(float Value, float BlackPoint, float WhitePoint)
{
    return clamp(BlackWhitePoint(Value, BlackPoint, WhitePoint), 0.0, 1.0);
}