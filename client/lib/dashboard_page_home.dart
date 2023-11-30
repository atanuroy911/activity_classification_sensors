import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:liquid_pull_to_refresh/liquid_pull_to_refresh.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'charts/line_chart.dart';
class DashboardPageHome extends StatefulWidget {
  const DashboardPageHome({super.key});

  @override
  _DashboardPageHomeState createState() => _DashboardPageHomeState();
}

class _DashboardPageHomeState extends State<DashboardPageHome> {
  final GlobalKey<LiquidPullToRefreshState> _refreshIndicatorKey =
  GlobalKey<LiquidPullToRefreshState>();

  String data = 'Fetched Data';
  String status = 'Server Connection Successful';
  String prediction = 'Awaiting Prediction ...';
  String roomData = 'Nothing';

  Stream<int> counterStream =
  Stream<int>.periodic(const Duration(seconds: 3), (x) => x);

  late String baseUrl; // To store the base URL
  late String port; // To store the port number

  Future<void> _loadSavedSettings() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    setState(() {
      baseUrl = prefs.getString('url') ?? ''; // Load saved URL
      port = prefs.getString('port') ?? ''; // Load saved port
    });
  }

  Future<void> fetchData() async {
    status = "Connecting...";
    final response =
    await http.get(Uri.parse('http://$baseUrl:$port/data'));
    if (response.statusCode == 200) {
      final decodedResponse = response.body;
      setState(() {
        data = decodedResponse;
        status = "Connection Succeeded";
      });
    }
    else {
      setState(() {
        status = "Server Connection FAILED";
      });
    }
    // print(data);
  }

  Future<void> fetchPrediction() async {
    final response = await http.post(
      Uri.parse('http://$baseUrl:$port/predict'),
      headers: {'Content-Type': 'application/json'},
      body: data,
    );

    if (response.statusCode == 200) {
      final decodedResponse = json.decode(response.body);
      final predictionValue = decodedResponse['prediction']; // Extract the 'prediction' value
      setState(() {
        prediction = predictionValue.toString();
      });
    }
  }

  Future<void> fetchRoomData() async {
    final response = await http.get(
        Uri.parse('http://$baseUrl:$port/room-data'));
    if (response.statusCode == 200) {
      final decodedResponse = response.body;
      // print(decodedResponse);
      setState(() {
        roomData = decodedResponse;
      });
    }
  }

  @override
  void initState() {
    super.initState();
    _refreshData();
  }

  Future<void> _refreshData() async {
    await _loadSavedSettings();
    await fetchData();
    await fetchPrediction();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: LiquidPullToRefresh(
        key: _refreshIndicatorKey,
        onRefresh: _refreshData,
        showChildOpacityTransition: false,
        child: ListView.builder(
          itemCount: 3, // Number of list items
          itemBuilder: (context, index) {
            if (index == 0) {
              return ChartPage(); // Your chart widget
            } else if (index == 1) {
              return ListTile(
                title: Text(
                  'Server Status',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                subtitle: Text(
                  status,
                ),
                leading: Icon(Icons.data_usage),
              );
            } else if (index == 2) {
              return ListTile(
                title: Text(
                  'Predicted Activity:',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                subtitle: Text(
                  prediction,
                ),
                leading: Icon(Icons.timeline),
              );
            } else {
              return StreamBuilder<int>(
                stream: counterStream,
                builder: (context, snapshot) {
                  return ListTile(
                    title: Text(
                      'Stream Value:',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    subtitle: Text(
                      'Here is the stream value received.',
                    ),
                    leading: Icon(Icons.stream),
                    trailing: Text('${snapshot.data ?? 'Loading...'}'),
                  );
                },
              );
            }
          },
        ),
      ),
    );
  }
}
