// Copyright MediaZ AS. All Rights Reserved.


#include "Track.h"

#include <mzFlatBuffersCommon.h>
#include <Args.h>
#include <Builtins_generated.h>

#include <mzUtil/Thread.h>
#include <mzFlatBuffersCommon.h>
#include <AppService_generated.h>

#include <asio.hpp>
#include <atomic>

#define trkMatrix          0x0000
#define trkEuler           0x0001

#define trkCameraToStudio  0x0000
#define trkStudioToCamera  0x0002

#define trkCameraZ_Up      0x0000
#define trkCameraY_Up      0x0004

#define trkStudioZ_Up      0x0000
#define trkStudioY_Up      0x0008

#define trkImageDistance   0x0000
#define trkFieldOfView     0x0010

#define trkHorizontal      0x0000
#define trkVertical        0x0020
#define trkDiagonal        0x0040

#define trkConsiderBlank   0x0000
#define trkIgnoreBlank     0x0080

#define trkAdjustAspect    0x0000
#define trkKeepAspect      0x0100

#define trkShiftOnChip     0x0000
#define trkShiftInPixels   0x0200

#define trkNetMagic           "DMC01"
#define trkNetHeaderType      6
#define trkNetHeaderFormat    7
#define trkNetHeaderSize      8


#define PI 3.14159265358979323846

using asio::ip::udp;
typedef uint8_t uint8;
typedef int8_t int8;
typedef uint32_t uint32;
typedef int32_t int32;

namespace mz
{
	typedef union
	{
		double         m[4][4];     /* m [j] [i] is matrix element */
									/* in row i, column j */
		struct
		{
			double   x;
			double   y;
			double   z;
			double   pan;
			double   tilt;
			double   roll;
		}           e;             /* "Euler" angles */

	} trkTransform;


	/* camera parameters */
	typedef struct
	{
		unsigned       id;            /* == 0 if not explicitly specified */

		unsigned       format;        /* bit mask of format options */

		trkTransform   t;             /* coordinate transformation */
		double         fov;           /* field of view or image distance */
		glm::dvec2	   center;       /* center shift */
		glm::dvec2	   k1k2;

		double         focdist;       /* depth of field simulation */
		double         aperture;

		unsigned long  counter;

	} trkCameraParams;

	/* camera constants */
	typedef struct
	{
		unsigned       id;            /* == 0 if not explicitly specified */

		int            imageWidth;
		int            imageHeight;
		int            blankLeft;
		int            blankRight;
		int            blankTop;
		int            blankBottom;

		double         chipWidth;
		double         chipHeight;
		double         fakeChipWidth;
		double         fakeChipHeight;
	} trkCameraConstants;

	struct XyncNodeContext : public TrackNodeContext
	{
	public:
		XyncNodeContext() :
			TrackNodeContext()
		{
		}
		

		int TrkStringToParams(const char* s, trkCameraParams* p) const
		{
			int n;

			n = sscanf_s(s, "I%u", &p->id);
			if (n != 1)
				p->id = 0;
			else if (!SkipWords(&s, 1))
				return 0;
			n = sscanf_s(s, "%x", &p->format);
			if (n != 1 || !SkipWords(&s, 1))   return 0;
			if (p->format & trkEuler)
			{
				n = sscanf_s(s, "%lg %lg %lg %lg %lg %lg",
					&p->t.e.x, &p->t.e.y, &p->t.e.z,
					&p->t.e.pan, &p->t.e.tilt, &p->t.e.roll);
				if (n != 6 || !SkipWords(&s, 6))   return 0;
			}
			else
			{
				n = sscanf_s(s, "%lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg",
					&p->t.m[0][0], &p->t.m[0][1], &p->t.m[0][2],
					&p->t.m[1][0], &p->t.m[1][1], &p->t.m[1][2],
					&p->t.m[2][0], &p->t.m[2][1], &p->t.m[2][2],
					&p->t.m[3][0], &p->t.m[3][1], &p->t.m[3][2]);
				if (n != 12 || !SkipWords(&s, 12))   return 0;
				/* assume that bottom row of matrix contains 0 0 0 1 */
				p->t.m[0][3] = p->t.m[1][3] = p->t.m[2][3] = 0.0;
				p->t.m[3][3] = 1.0;
			}
			n = sscanf_s(s, "%lg %lg %lg %lg %lg %lg %lg %lu",
				&p->fov, &p->center.x, &p->center.y, &p->k1k2.x, &p->k1k2.y,
				&p->focdist, &p->aperture, &p->counter);
			if (n != 8 || !SkipWords(&s, 8))   return 0;
			while (isspace(*s))   ++s;
			if (*s != '\0')   return 0;
			return 1;

		}

		int TrkStringToConstants(const char* s, trkCameraConstants* c) const
		{
			int n;

			n = sscanf_s(s, "I%u", &c->id);
			if (n != 1)
				c->id = 0;
			else if (!SkipWords(&s, 1))
				return 0;
			n = sscanf_s(s, "%i %i %i %i %i %i %lg %lg %lg %lg",
				&c->imageWidth, &c->imageHeight,
				&c->blankLeft, &c->blankRight,
				&c->blankTop, &c->blankBottom,
				&c->chipWidth, &c->chipHeight,
				&c->fakeChipWidth, &c->fakeChipHeight);
			if (n != 10 || !SkipWords(&s, 10))   return 0;
			while (isspace(*s))   ++s;
			if (*s != '\0')   return 0;
			return 1;
		}

		int SkipWords(const char** string, int numWords) const
		{
			const char* s = *string;
			int wordCount;

			for (wordCount = 0; wordCount < numWords; ++wordCount)
			{
				while (isspace(*s))   ++s;
				if (*s == '\0')   break;
				while (*s != '\0' && !isspace(*s))   ++s;
			}
			*string = s;
			return wordCount == numWords;
		}


        std::vector<u8> ParseableBytes(std::vector<u8> buffer)
        {
			for (int32 Index = 0; Index < buffer.size(); ++Index)
			{
				if (buffer[Index] == 'D')
				{
					std::string MagicText(buffer.begin() + Index, buffer.begin() + Index + 5);
					if (MagicText == trkNetMagic)
					{
						buffer.erase(buffer.begin(), buffer.begin()+Index);
					}
				}
			}
            return buffer;
        }

        bool Parse(std::vector<u8> const& data, fb::TTrack& TrackData) override
		{
            TrackData.sensor_size = fb::vec2d(9.590, 5.394);
            TrackData.fov = 60;
            TrackData.distortion_scale = 1;
            TrackData.pixel_aspect_ratio = 1;
            
            auto buffer = ParseableBytes(data);
			auto NumBytes = buffer.size();
        
			trkCameraConstants   CameraConstants;
			trkCameraParams      CameraParameters;

			if (NumBytes > trkNetHeaderSize)
			{
				//FString Line = ANSI_TO_TCHAR((char *)&Buffer[0]);

				auto typeChar = buffer[trkNetHeaderType];
				auto formatChar = buffer[trkNetHeaderFormat];

				int parameterserrorhandler = 0;
				int constantserrorhandler = 0;

				if (typeChar == 'C')
				{
					/* datagram contains camera constants */
					if (formatChar == 'B')
					{
						/* binary format */
						//NOT IMPLEMENTED
					}
					else
					{
						/* ASCII format */
						constantserrorhandler = TrkStringToConstants((char*)&buffer[trkNetHeaderSize], &CameraConstants);
					}
				}
				else
				{
					/* datagram contains camera parameters */
					if (formatChar == 'B')
					{
						/* binary format */
						//NOT IMPLEMENTED
					}
					else
					{
						/* ASCII format */
						parameterserrorhandler = TrkStringToParams((char*)&buffer[trkNetHeaderSize], &CameraParameters);
					}
				}
				if (!constantserrorhandler && !parameterserrorhandler)
				{
					mzEngine.LogE("Cannot parse the bytes sent with size %d at xync track node", NumBytes);
				}
				if (constantserrorhandler)
				{
                    TrackData.sensor_size = fb::vec2d(CameraConstants.chipWidth, CameraConstants.chipHeight);
					buffer.clear();
					NumBytes = buffer.size();
				}

				if (parameterserrorhandler)
				{
					//check if format has Matrix
					if (CameraParameters.format & trkEuler)
					{
						auto posY = (CameraParameters.format & trkStudioY_Up) ? CameraParameters.t.e.z : CameraParameters.t.e.y;
						auto posZ = (CameraParameters.format & trkStudioY_Up) ? -CameraParameters.t.e.y : CameraParameters.t.e.z;
                        TrackData.location = fb::vec3d(
                            CameraParameters.t.e.x * 100.0,
                            posY * 100,
                            posZ * 100
                        );
                        TrackData.rotation = fb::vec3d(
                            CameraParameters.t.e.tilt,
                            CameraParameters.t.e.pan,
                            CameraParameters.t.e.roll
                        );
					}
					else
					{
						double      temp, shiftX, shiftY, shiftZ;
						if (CameraParameters.format & trkCameraY_Up)
						{
							/* exchange middle columns, then negate entries in third column */
							temp = CameraParameters.t.m[2][0];   CameraParameters.t.m[2][0] = -CameraParameters.t.m[1][0];   CameraParameters.t.m[1][0] = temp;
							temp = CameraParameters.t.m[2][1];   CameraParameters.t.m[2][1] = -CameraParameters.t.m[1][1];   CameraParameters.t.m[1][1] = temp;
							temp = CameraParameters.t.m[2][2];   CameraParameters.t.m[2][2] = -CameraParameters.t.m[1][2];   CameraParameters.t.m[1][2] = temp;
						}

						if (CameraParameters.format & trkStudioY_Up)
						{
							/* exchange middle rows, then negate entries in third row */
							temp = CameraParameters.t.m[0][2];   CameraParameters.t.m[0][2] = -CameraParameters.t.m[0][1];   CameraParameters.t.m[0][1] = temp;
							temp = CameraParameters.t.m[1][2];   CameraParameters.t.m[1][2] = -CameraParameters.t.m[1][1];   CameraParameters.t.m[1][1] = temp;
							temp = CameraParameters.t.m[2][2];   CameraParameters.t.m[2][2] = -CameraParameters.t.m[2][1];   CameraParameters.t.m[2][1] = temp;
							temp = CameraParameters.t.m[3][2];   CameraParameters.t.m[3][2] = -CameraParameters.t.m[3][1];   CameraParameters.t.m[3][1] = temp;
						}

						if (CameraParameters.format & trkStudioToCamera)
						{
							/*
							* invert matrix:
							* since upper left 3x3 matrix is orthogonal, its inverse
							* is calculated by transposing it;
							* the new translation vector in the fourth column is calculated
							* by applying the previous translation vector to the inverted
							* upper left 3x3 matrix
							*/
							temp = CameraParameters.t.m[0][1];   CameraParameters.t.m[0][1] = CameraParameters.t.m[1][0];   CameraParameters.t.m[1][0] = temp;
							temp = CameraParameters.t.m[0][2];   CameraParameters.t.m[0][2] = CameraParameters.t.m[2][0];   CameraParameters.t.m[2][0] = temp;
							temp = CameraParameters.t.m[1][2];   CameraParameters.t.m[1][2] = CameraParameters.t.m[2][1];   CameraParameters.t.m[2][1] = temp;
							shiftX = -(CameraParameters.t.m[0][0] * CameraParameters.t.m[3][0] + CameraParameters.t.m[1][0] * CameraParameters.t.m[3][1] + CameraParameters.t.m[2][0] * CameraParameters.t.m[3][2]);
							shiftY = -(CameraParameters.t.m[0][1] * CameraParameters.t.m[3][0] + CameraParameters.t.m[1][1] * CameraParameters.t.m[3][1] + CameraParameters.t.m[2][1] * CameraParameters.t.m[3][2]);
							shiftZ = -(CameraParameters.t.m[0][2] * CameraParameters.t.m[3][0] + CameraParameters.t.m[1][2] * CameraParameters.t.m[3][1] + CameraParameters.t.m[2][2] * CameraParameters.t.m[3][2]);
							CameraParameters.t.m[3][0] = shiftX;
							CameraParameters.t.m[3][1] = shiftY;
							CameraParameters.t.m[3][2] = shiftZ;
						}

						//RotationMatrix to Euler conversion.
						float sy = sqrt(CameraParameters.t.m[0][0] * CameraParameters.t.m[0][0] + CameraParameters.t.m[0][1] * CameraParameters.t.m[0][1]);
						bool singular = sy < 1e-6; // If

						float x, y, z;
						if (!singular)
						{
							x = atan2(CameraParameters.t.m[1][2], CameraParameters.t.m[2][2]);
							y = atan2(-CameraParameters.t.m[0][2], sy);
							z = atan2(CameraParameters.t.m[0][1], CameraParameters.t.m[0][0]);
						}
						else
						{
							x = atan2(-CameraParameters.t.m[2][1], CameraParameters.t.m[1][1]);
							y = atan2(-CameraParameters.t.m[0][2], sy);
							z = 0;
						}

						auto degreesX = x * (180.0 / PI);
						auto degreesY = y * (180.0 / PI);
						auto degreesZ = z * (180.0 / PI) + 90.0;

						auto posY = (CameraParameters.format & trkStudioY_Up) ? CameraParameters.t.m[3][2] : CameraParameters.t.m[3][1];
						auto posZ = (CameraParameters.format & trkStudioY_Up) ? -CameraParameters.t.m[3][1] : CameraParameters.t.m[3][2];

						TrackData.location  = fb::vec3d(CameraParameters.t.m[3][0] * 100, posY * 100, posZ * 100);
                        TrackData.rotation = fb::vec3d(degreesX, degreesY, degreesZ);
					}

					if (CameraParameters.format & trkFieldOfView)
					{
						if (CameraParameters.format & trkVertical)
						{
							auto radFov = CameraParameters.fov * PI / 180.0;
							auto radResult = 2 * atan(tan((float)radFov / 2) * 1.77777F); //TODO chipsize - aspect ratio
							TrackData.fov = radResult * (180.0 / PI);
						}
						else if (CameraParameters.format & trkHorizontal)
						{
							TrackData.fov = CameraParameters.fov;
						}
						else
						{
							TrackData.fov = CameraParameters.fov;
						}
					}
					else
					{
						TrackData.fov = atan(0.5 * 1920 / CameraParameters.fov) * (360.0 / 3.14);
					}

					if (CameraParameters.format & trkIgnoreBlank)
					{
						if (CameraParameters.format & trkAdjustAspect)
						{

						}
						else
						{
							//trkKeepAspect
						}
					}
					else
					{
						//trkConsiderblank
					}

                    (glm::dvec2&)TrackData.center_shift = CameraParameters.center;

					if (CameraParameters.format & trkShiftInPixels)
					{
						(glm::dvec2&)TrackData.center_shift *= glm::dvec2(9.590, 5.394) / glm::dvec2(1920, 1080);
					}

					TrackData.focus = 1;
					TrackData.zoom = 1;
	                TrackData.focus_distance = CameraParameters.focdist * 100;

					auto SensorAspectRatio = TrackData.sensor_size.x() / TrackData.sensor_size.y();
					auto SensorHalfWidth = TrackData.sensor_size.x() / 2.0f;
					auto RD2 = SensorHalfWidth * SensorHalfWidth + (SensorHalfWidth / SensorAspectRatio) * (SensorHalfWidth / SensorAspectRatio);
					auto RD4 = RD2 * RD2;
					CameraParameters.k1k2 *= glm::dvec2(RD2, RD4);
					(glm::dvec2&)TrackData.k1k2 = CameraParameters.k1k2;
					buffer.clear();
					NumBytes = buffer.size();
				}
			}
		
			return true;
		}

		~XyncNodeContext() 
		{
			if (IsRunning())
			{
				Stop();
			}
		}

		//mz::fb::Track* outTrack;
	};

void RegisterXyncNode(mzNodeFunctions& functions)
{
	functions.TypeName = "mz.track.Xync";
	RegisterTrackCommon(functions);

    functions.OnNodeCreated = [](fb::Node const* node, void** ctx) {
        auto context = new XyncNodeContext();
        *ctx = context;
        auto pins = context->Load(*node);
		if (auto pin = pins[UDP_Port_Name])
		{
			if (flatbuffers::IsFieldPresent(pin, fb::Pin::VT_DATA))
			{
				context->Port = *(uint16_t*)pin->data()->data();
			}
		}
		if (auto pin = pins[Enable_Name])
		{
			if (flatbuffers::IsFieldPresent(pin, fb::Pin::VT_DATA))
			{
				if (*(bool*)pin->data()->data())
				{
					context->Start();
				}
			}
		}
		//context->Start();
    };

	functions.OnPinValueChanged = [](auto ctx, auto id, mzBuffer* value) 
	{ 
		XyncNodeContext* fnctx = (XyncNodeContext*)ctx;
		fnctx->OnPinValueChanged(id, value->Data);
	};

	functions.OnNodeDeleted = [](auto ctx, auto id) {
		delete (XyncNodeContext*)ctx;
	};

}

} // namespace mz