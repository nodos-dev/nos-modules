// Copyright MediaZ AS. All Rights Reserved.

#include "MediaZ/Helpers.hpp"
#include "Builtins_generated.h"
#include "glm/gtx/quaternion.hpp"
#include "glm/glm.hpp"

namespace mz
{

void RegisterController(MzNodeFunctions& functions)
{
    functions.TypeName = "mz.track.UserTrack";
    struct UserTrack {
         //union
         //{
         //   struct
         //   {
         //       glm::dvec3 r;
         //       glm::dvec3 u;
         //       glm::dvec3 f;
         //   };
         //   glm::dmat3 ori;
         //};

        // Controller() : r(0, 1, 0), u(0, 0, 1), f(1, 0, 0) {}
        
        glm::dvec3 p   = glm::vec3(0, 0, 5);
        f64 yaw = 0;
        f64 pitch = 0;
        glm::dvec3 v{};

        f64 impulse = 1.0;
        f64 decay = 0;

        mz::fb::UUID OutPinId;
        f64 fov = 90.0;
        bool zoom = false;

        void fill(fb::TTrack& track)
        {
            // Assumes fields are present in the table.
            auto& rot = track.rotation;
            rot.mutate_y(glm::degrees(pitch));
            rot.mutate_z(glm::degrees(yaw));
            rot.mutate_x(0);
            auto& loc = track.location;
            loc.mutate_x(p.x);
            loc.mutate_y(p.y);
            loc.mutate_z(p.z);
            track.fov = zoom ? 30.f : 74.f;
            track.distortion_scale = 1;
            track.render_ratio = 1;
            track.pixel_aspect_ratio = 1;
        	track.sensor_size = fb::vec2d(9.590, 5.394);
        	auto val = mz::Buffer::From(track);
        	mzEngine.SetPinValue(OutPinId, {.Data = val.data(), .Size = val.size()});
        }
    };

    functions.OnNodeCreated = [](fb::Node const* n, void** ctx) {
        auto c = new UserTrack();
        *ctx = c;
        c->OutPinId = *n->pins()->begin()->id();
        auto trackBuf = mz::Buffer(n->pins()->begin()->data());
        auto track = trackBuf.As<mz::fb::TTrack>();
        c->p = (glm::dvec3&)(track.location);
        auto rot = (glm::dvec3&)(track.rotation);
        c->pitch  = glm::radians(rot.y);
        c->yaw    = glm::radians(rot.z);
        c->fov    = track.fov;
        c->fill(track);
        //flatbuffers::FlatBufferBuilder fbb;
        //std::vector<flatbuffers::Offset<mz::fb::NodeStatusMessage>> msg;
        //msg.push_back(fb::CreateNodeStatusMessageDirect(fbb, "Hold RMB to start controlling", fb::NodeStatusMessageType::INFO));
        //msg.push_back(fb::CreateNodeStatusMessageDirect(fbb, "Use your mouse to look around", fb::NodeStatusMessageType::INFO));
        //msg.push_back(fb::CreateNodeStatusMessageDirect(fbb, "WASD to move", fb::NodeStatusMessageType::INFO));
        //msg.push_back(fb::CreateNodeStatusMessageDirect(fbb, "Space to reset view", fb::NodeStatusMessageType::INFO));
        //msg.push_back(fb::CreateNodeStatusMessageDirect(fbb, "Good luck have fun", fb::NodeStatusMessageType::INFO));
        //services.HandleEvent(CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, n.id(), ClearFlags::NONE, 0, 0, 0, 0, 0, 0, &msg)));
    };

    functions.OnNodeDeleted = [](void* ctx, auto){delete (UserTrack*)ctx;};
    functions.OnKeyEvent = [](void* ctx, MzKeyEvent const* keyEvent){
		auto& key = keyEvent->Key;
    	auto& mdelta = keyEvent->MouseDelta;
    	auto c = (UserTrack*)ctx;
        // c->fov += 2 *(((key == 'F') - (key == 'G')));
        // c->m *= 1.0 + 0.2 * ((key == 'O') - (key == 'P'));
        if (32 == key)
        {
            c->pitch = 0;
            c->yaw   = 180;
            c->p = glm::vec3{600, 50, 100};
            flatbuffers::FlatBufferBuilder fbb;
            mzEngine.HandleEvent(CreateAppEvent(fbb, mz::app::CreatePinDirtied(fbb, &c->OutPinId)));
        }

        glm::dvec2 ANG = glm::dvec2(c->yaw, c->pitch);
        glm::dvec2 COS = cos(ANG);
        glm::dvec2 SIN = sin(ANG);

        glm::dvec3 f = glm::dvec3(COS.y * COS.x, COS.y * SIN.x, SIN.y);
        glm::dvec3 r = glm::dvec3(-SIN.x, COS.x, 0);
        glm::dvec3 u = cross(f, r);
        
        glm::dvec3 pf =  f * f64((key == 'W') - (key == 'S'));
        glm::dvec3 pr =  r * f64((key == 'D') - (key == 'A'));
        glm::dvec3 pu =  u * f64((key == 'E') - (key == 'Q'));
        
        c->v += c->impulse * (pf + pr + pu);
        c->p += 10.0 * (pf + pr + pu);

        // c->zoom ^= (action != 2 && button == 1);
        c->pitch -= mdelta.y * 0.015;
        c->yaw   += mdelta.x * 0.015;
        
        c->yaw   = fmod(c->yaw, glm::radians(360.));
        // c->pitch = glm::clamp(c->pitch, glm::radians(-80.), glm::radians(80.));
        
        flatbuffers::FlatBufferBuilder fbb;
        mzEngine.HandleEvent(CreateAppEvent(fbb, mz::app::CreatePinDirtied(fbb, &c->OutPinId)));
    };

    functions.ExecuteNode = [](void* ctx, const MzNodeExecuteArgs* args){
        auto c = (UserTrack*)ctx;
    	auto values = GetPinValues(args);
        c->impulse = glm::max(*(f64*)values["Impulse"], 1.);
        c->decay   = glm::max(*(f64*)values["Decay"], 0.);
        c->p += c->v;
        c->v *= exp(-c->decay);
    	auto track = GetPinValue<fb::TTrack>(values, "Track");
        c->fill(track);
        return MZ_RESULT_SUCCESS;
    };
}

}