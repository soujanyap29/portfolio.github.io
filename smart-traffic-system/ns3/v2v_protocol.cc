// NS3 V2V Communication Implementation
// Part of Smart Traffic Management System

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("V2VProtocol");

class V2VApplication : public Application {
public:
    V2VApplication();
    virtual ~V2VApplication();
    
    void Setup(Ptr<Socket> socket, Address address, uint32_t packetSize, 
               uint32_t nPackets, DataRate dataRate);

private:
    virtual void StartApplication(void);
    virtual void StopApplication(void);
    
    void ScheduleTx(void);
    void SendPacket(void);
    
    Ptr<Socket> m_socket;
    Address m_peer;
    uint32_t m_packetSize;
    uint32_t m_nPackets;
    DataRate m_dataRate;
    EventId m_sendEvent;
    bool m_running;
    uint32_t m_packetsSent;
};

V2VApplication::V2VApplication()
    : m_socket(0),
      m_peer(),
      m_packetSize(0),
      m_nPackets(0),
      m_dataRate(0),
      m_sendEvent(),
      m_running(false),
      m_packetsSent(0)
{
}

V2VApplication::~V2VApplication()
{
    m_socket = 0;
}

void V2VApplication::Setup(Ptr<Socket> socket, Address address, 
                           uint32_t packetSize, uint32_t nPackets, 
                           DataRate dataRate)
{
    m_socket = socket;
    m_peer = address;
    m_packetSize = packetSize;
    m_nPackets = nPackets;
    m_dataRate = dataRate;
}

void V2VApplication::StartApplication(void)
{
    m_running = true;
    m_packetsSent = 0;
    m_socket->Bind();
    m_socket->Connect(m_peer);
    SendPacket();
}

void V2VApplication::StopApplication(void)
{
    m_running = false;
    
    if (m_sendEvent.IsRunning())
    {
        Simulator::Cancel(m_sendEvent);
    }
    
    if (m_socket)
    {
        m_socket->Close();
    }
}

void V2VApplication::SendPacket(void)
{
    // Create V2V message packet
    Ptr<Packet> packet = Create<Packet>(m_packetSize);
    
    // Add headers (position, speed, etc.)
    // In a real implementation, these would be actual data structures
    
    m_socket->Send(packet);
    m_packetsSent++;
    
    NS_LOG_INFO("Packet sent at " << Simulator::Now().GetSeconds() << "s");
    
    if (m_packetsSent < m_nPackets)
    {
        ScheduleTx();
    }
}

void V2VApplication::ScheduleTx(void)
{
    if (m_running)
    {
        Time tNext(Seconds(m_packetSize * 8 / static_cast<double>(m_dataRate.GetBitRate())));
        m_sendEvent = Simulator::Schedule(tNext, &V2VApplication::SendPacket, this);
    }
}

// Main simulation setup
int main(int argc, char *argv[])
{
    // Command line parameters
    uint32_t nVehicles = 50;
    uint32_t nRSUs = 4;
    double simTime = 100.0;
    bool verbose = false;
    
    CommandLine cmd;
    cmd.AddValue("nVehicles", "Number of vehicle nodes", nVehicles);
    cmd.AddValue("nRSUs", "Number of RSU nodes", nRSUs);
    cmd.AddValue("simTime", "Total simulation time (seconds)", simTime);
    cmd.AddValue("verbose", "Enable verbose logging", verbose);
    cmd.Parse(argc, argv);
    
    if (verbose)
    {
        LogComponentEnable("V2VProtocol", LOG_LEVEL_INFO);
    }
    
    NS_LOG_INFO("Creating " << nVehicles << " vehicle nodes and " << nRSUs << " RSU nodes");
    
    // Create nodes
    NodeContainer vehicleNodes;
    vehicleNodes.Create(nVehicles);
    
    NodeContainer rsuNodes;
    rsuNodes.Create(nRSUs);
    
    // Setup WAVE (802.11p) for V2V communication
    YansWifiChannelHelper waveChannel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper wavePhy = YansWifiPhyHelper::Default();
    wavePhy.SetChannel(waveChannel.Create());
    
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211p);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode", StringValue("OfdmRate6MbpsBW10MHz"),
                                 "ControlMode", StringValue("OfdmRate6MbpsBW10MHz"));
    
    // MAC layer setup
    WifiMacHelper waveMac;
    waveMac.SetType("ns3::OcbWifiMac");
    
    // Install devices
    NetDeviceContainer vehicleDevices = wifi.Install(wavePhy, waveMac, vehicleNodes);
    NetDeviceContainer rsuDevices = wifi.Install(wavePhy, waveMac, rsuNodes);
    
    // Mobility model for vehicles
    MobilityHelper vehicleMobility;
    vehicleMobility.SetPositionAllocator("ns3::RandomRectanglePositionAllocator",
                                        "X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=2000.0]"),
                                        "Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=2000.0]"));
    
    vehicleMobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
    vehicleMobility.Install(vehicleNodes);
    
    // Set random velocities for vehicles
    for (uint32_t i = 0; i < vehicleNodes.GetN(); i++)
    {
        Ptr<ConstantVelocityMobilityModel> mob = vehicleNodes.Get(i)->GetObject<ConstantVelocityMobilityModel>();
        mob->SetVelocity(Vector(10.0, 0.0, 0.0));  // 10 m/s in X direction
    }
    
    // Mobility model for RSUs (stationary)
    MobilityHelper rsuMobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
    positionAlloc->Add(Vector(500.0, 500.0, 0.0));
    positionAlloc->Add(Vector(1500.0, 500.0, 0.0));
    positionAlloc->Add(Vector(500.0, 1500.0, 0.0));
    positionAlloc->Add(Vector(1500.0, 1500.0, 0.0));
    
    rsuMobility.SetPositionAllocator(positionAlloc);
    rsuMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    rsuMobility.Install(rsuNodes);
    
    // Internet stack
    InternetStackHelper internet;
    internet.Install(vehicleNodes);
    internet.Install(rsuNodes);
    
    // Assign IP addresses
    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer vehicleInterfaces = ipv4.Assign(vehicleDevices);
    Ipv4InterfaceContainer rsuInterfaces = ipv4.Assign(rsuDevices);
    
    // Setup V2V communication applications
    TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");
    
    for (uint32_t i = 0; i < vehicleNodes.GetN(); i++)
    {
        Ptr<Socket> recvSocket = Socket::CreateSocket(vehicleNodes.Get(i), tid);
        InetSocketAddress local = InetSocketAddress(Ipv4Address::GetAny(), 8080);
        recvSocket->Bind(local);
        recvSocket->SetRecvCallback(MakeNullCallback<void, Ptr<Socket>>());
        
        // Broadcast messages periodically
        Ptr<Socket> sendSocket = Socket::CreateSocket(vehicleNodes.Get(i), tid);
        InetSocketAddress remote = InetSocketAddress(Ipv4Address("255.255.255.255"), 8080);
        sendSocket->SetAllowBroadcast(true);
        
        Ptr<V2VApplication> app = CreateObject<V2VApplication>();
        app->Setup(sendSocket, remote, 512, 1000, DataRate("1Mbps"));
        vehicleNodes.Get(i)->AddApplication(app);
        app->SetStartTime(Seconds(1.0 + i * 0.01));
        app->SetStopTime(Seconds(simTime));
    }
    
    // Enable packet capture
    wavePhy.EnablePcapAll("v2v-communication");
    
    // Run simulation
    NS_LOG_INFO("Starting simulation for " << simTime << " seconds");
    Simulator::Stop(Seconds(simTime));
    Simulator::Run();
    Simulator::Destroy();
    
    NS_LOG_INFO("Simulation completed");
    
    return 0;
}
