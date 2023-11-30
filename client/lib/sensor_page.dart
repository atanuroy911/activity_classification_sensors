import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'package:http/http.dart' as http;
import 'package:flutter/material.dart';
import 'package:liquid_pull_to_refresh/liquid_pull_to_refresh.dart';
import 'package:shared_preferences/shared_preferences.dart';


class SensorPage extends StatelessWidget {
  const SensorPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const _SensorPage(title: 'Liquid Pull To Refresh'),
    );
  }
}

class _SensorPage extends StatefulWidget {
  const _SensorPage({Key? key, this.title}) : super(key: key);

  final String? title;

  @override
  _SensorPageState createState() => _SensorPageState();
}

class _SensorPageState extends State<_SensorPage> {
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();
  final GlobalKey<LiquidPullToRefreshState> _refreshIndicatorKey =
  GlobalKey<LiquidPullToRefreshState>();

  String data = 'Fetched Data';

  ScrollController? _scrollController;
  late String baseUrl; // To store the base URL
  late String port; // To store the port number

  @override
  void initState() {
    super.initState();
    _handleRefresh();
    _scrollController = ScrollController();
  }


  static final List<String> _items = <String>[
    'D001', 'D002', 'D004', 'M001', 'M002', 'M003', 'M004', 'M005', 'M006', 'M007', 'M008', 'M009', 'M010', 'M011', 'M012', 'M013', 'M014', 'M015', 'M016', 'M017', 'M018', 'M019', 'M020', 'M021', 'M022', 'M023', 'M024', 'M025', 'M026', 'M027', 'M028', 'M029', 'M030', 'M031'
  ];

  Future<void> _loadSavedSettings() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    setState(() {
      baseUrl = prefs.getString('url') ?? ''; // Load saved URL
      port = prefs.getString('port') ?? ''; // Load saved port
    });
  }

  Future<String> fetchData() async {
    final response =
    await http.get(Uri.parse('http://$baseUrl:$port/last-data'));
    if (response.statusCode == 200) {
      final decodedResponse = response.body;
      return decodedResponse;
    } else {
      return "Server Connection FAILED";
    }
  }

  Future<void> _handleRefresh() async {
    await _loadSavedSettings();
    final fetchedData = await fetchData();
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: const Text('Refresh complete'),
        action: SnackBarAction(
          label: 'OK',
          onPressed: () {},
        ),
      ),
    );
    setState(() {
      data = fetchedData;
    });
  }


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      key: _scaffoldKey,
      body: LiquidPullToRefresh(
        key: _refreshIndicatorKey,
        onRefresh: _handleRefresh,
        showChildOpacityTransition: false,
        child: FutureBuilder(
          future: fetchData(),
          builder: (context, snapshotData) {
            if (snapshotData.connectionState == ConnectionState.waiting) {
              return Center(child: CircularProgressIndicator());
            } else if (snapshotData.hasError) {
              return Center(child: Text('Error: ${snapshotData.error}'));
            } else if (snapshotData.hasData) {
              final jsonData = jsonDecode(snapshotData.data.toString());
              final data = jsonData['data'][0].sublist(1); // Exclude the time

              return ListView.builder(
                padding: kMaterialListPadding,
                itemCount: _items.length,
                controller: _scrollController,
                itemBuilder: (BuildContext context, int index) {
                  final String item = _items[index];
                  final sensorState = data[index];
                  final Color avatarColor = sensorState == 'ON' ? Colors.green : Colors.red;

                  return ListTile(
                    isThreeLine: true,
                    leading: CircleAvatar(
                      radius: 30,
                      backgroundColor: avatarColor,
                      child: Text(item),
                    ),
                    title: Text('Sensor $item.'),
                    subtitle: Text('State: $sensorState'),
                  );
                },
              );
            } else {
              return Center(child: Text('No data available.'));
            }
          },
        ),
      ),
    );
  }
}
