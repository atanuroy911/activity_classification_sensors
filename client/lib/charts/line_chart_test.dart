import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Line Chart Sample',
      home: LineChartSample(),
    );
  }
}

class LineChartSample extends StatefulWidget {
  @override
  _LineChartSampleState createState() => _LineChartSampleState();
}

class _LineChartSampleState extends State<LineChartSample> {
  Map<String, List<List<String>>> rawData = {};

  @override
  void initState() {
    super.initState();
    fetchData(); // Call fetchData instead of fetchRoomData directly
  }

  Future<void> fetchData() async {
    await fetchRoomData(); // Use await to wait for the fetchRoomData to complete
  }

  Future<void> fetchRoomData() async {
    final response = await http.get(
        Uri.parse('http://172.27.21.64:12345/room-data'));
    if (response.statusCode == 200) {
      final decodedResponse = response.body;
      // print(decodedResponse);
      setState(() {
        rawData = parseResponse(decodedResponse);
      });
    }
  }

  Map<String, List<List<String>>> parseResponse(String response) {
    final parsedData = Map<String, List<List<String>>>();

    final decodedJson = jsonDecode(response);

    if (decodedJson.containsKey('bedroom')) {
      final bedroomData = decodedJson['bedroom'];
      if (kDebugMode) {
        print(bedroomData);
      }

      List<List<String>> parsedBedroomData = [];
      for (var entry in bedroomData) {
        if (entry is List) {
          List<String> parsedEntry = entry.map((e) => e.toString()).toList();
          parsedBedroomData.add(parsedEntry);
        }
      }

      parsedData['bedroom'] = parsedBedroomData;
    }

    return parsedData;
  }

  @override
  Widget build(BuildContext context) {
    // List<List<FlSpot>> spotsLists = generateSpotsLists(rawData); // Generate spotsLists here

    return Scaffold(
      appBar: AppBar(
        title: Text('Line Chart Sample'),
      ),
      body: LineChart(
        LineChartData(
          gridData: FlGridData(show: false),
          titlesData: FlTitlesData(show: false),
          borderData: FlBorderData(show: false),
          lineBarsData: generateLineBarsData(), // Pass spotsLists here
        ),
      ),
    );
  }
  double convertTimeStringToDouble(String timeString) {
    List<String> parts = timeString.split(':');
    int hours = int.parse(parts[0]);
    int minutes = int.parse(parts[1]);

    List<String> secondsAndMilliseconds = parts[2].split('.');
    int seconds = int.parse(secondsAndMilliseconds[0]);
    int milliseconds = int.parse(secondsAndMilliseconds[1]);

    double totalSeconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000;
    return totalSeconds;
  }

  Map<String, String> separateParts(String input) {
    String cleanedInput = input.replaceAll("[", "").replaceAll("]", "");
    List<String> parts = cleanedInput.split(',').map((part) => part.trim()).toList();

    String time = parts[0];
    Map<String, String> statusMap = {};

    for (int i = 1; i < parts.length; i++) {
      statusMap['Status ${i}'] = parts[i] == 'ON' ? '1' : '0';
    }
    double timeInSeconds = convertTimeStringToDouble(time);
    Map<String, String> partsMap = {
      'Time': timeInSeconds.toString(),
      ...statusMap,
    };

    return partsMap;
  }

  List<LineChartBarData> generateLineBarsData() {
    final List<LineChartBarData> lineBarsData = [];
    List<List<FlSpot>> spotsLists = [];

    rawData.forEach((key, value) {
      print(key);
      print(value);
      Map<String, String> parts = separateParts(value[0][0]);
      List<String> keys = parts.keys.toList();
      for (int j = 1; j < keys.length; j++) {
        List<FlSpot> spots = [];
        for (int k = 0; k < value.length; k++) {
          Map<String, String> parts = separateParts(value[k][0]);
          spots.add(FlSpot(
            double.parse(parts[keys[0]]!),
            double.parse(parts[keys[j]]!),
          ));
        }
        spotsLists.add(spots);
      }

    });

    for (var spotsList in spotsLists) {
      LineChartBarData lineChartBarData = LineChartBarData(
        spots: spotsList,
        isCurved: true,
        dotData: const FlDotData(show: false),
        belowBarData: BarAreaData(show: false),
      );

      lineBarsData.add(lineChartBarData);
    }

    return lineBarsData;
  }
}
