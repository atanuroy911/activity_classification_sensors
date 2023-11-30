import 'dart:convert';

import 'package:d_chart/d_chart.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';


class ChartPage extends StatefulWidget {
  // final String roomData;

  const ChartPage({super.key});

  @override
  _ChartPageState createState() => _ChartPageState();
}

class _ChartPageState extends State<ChartPage> {
  List<Map<String, dynamic>> rawData = [];
  late String baseUrl; // To store the base URL
  late String port; // To store the port number
  Future<void> _loadSavedSettings() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    setState(() {
      baseUrl = prefs.getString('url') ?? ''; // Load saved URL
      port = prefs.getString('port') ?? ''; // Load saved port
    });
  }
  Future<void> fetchRoomData() async {
    await _loadSavedSettings();
    final response = await http.get(
        Uri.parse('http://$baseUrl:$port/room-data'));
    if (response.statusCode == 200) {
      final decodedResponse = response.body;
      // print(decodedResponse);
      setState(() {
        rawData = parseResponse(decodedResponse);
      });
    }
  }

  double convertTimeStringToDouble(String timeString) {
    List<String> parts = timeString.split(':');
    int hours = int.parse(parts[0]);
    int minutes = int.parse(parts[1]);

    List<String> secondsAndMilliseconds = parts[2].split('.');
    int seconds = int.parse(secondsAndMilliseconds[0]);
    int milliseconds = int.parse(secondsAndMilliseconds[1]);

    double totalSeconds = hours * 3600 + minutes * 60 + seconds +
        milliseconds / 1000;
    return scaleTimeToRange(totalSeconds);
  }

  double scaleTimeToRange(double timeValue) {
    // Assuming the original time values are in seconds
    double scaledValue = (timeValue / (24 * 3600)) *
        10; // Scale to a range from 0 to 10
    return scaledValue;
  }

  List<Map<String, dynamic>> parseResponse(String response) {
    final decodedJson = jsonDecode(response);
    final parsedList = <Map<String, dynamic>>[];

    decodedJson.forEach((key, bedroomData) {
      for (int i = 1; i < bedroomData[0][0].length; i++) {
        final id = 'Line $i';
        final data = 'data';

        List<Map<String, dynamic>> parsedEntryData = [];

        for (int j = 0; j < bedroomData.length; j++) {
          final domain = j;
          final measure = bedroomData[j][0][i];

          parsedEntryData.add({
            // 'domain': convertTimeStringToDouble(domain).toDouble(),
            'domain': domain.toDouble(),
            'measure': measure == 'OFF' ? 0 + i : 1 + i,
          });
        }

        final parsedData = {
          'id': id,
          data: parsedEntryData,
        };

        parsedList.add(parsedData);
      }
    });

    // print(parsedList);
    return parsedList;
  }


  Map<String, String> separateParts(String input) {
    String cleanedInput = input.replaceAll("[", "").replaceAll("]", "");
    List<String> parts = cleanedInput.split(',')
        .map((part) => part.trim())
        .toList();

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

  @override
  void initState() {
    super.initState();
    fetchData(); // Call fetchData instead of fetchRoomData directly
  }

  Future<void> fetchData() async {
    await fetchRoomData(); // Use await to wait for the fetchRoomData to complete
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.all(16),
      child: Column(
        children: [
          // Title Box
          Container(
            padding: EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: Colors.grey[300],
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              'Summary Data',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
          ),
          SizedBox(height: 16), // Spacer

          // DChartLine Widget
          AspectRatio(
            aspectRatio: 16 / 9,
            child: DChartLine(
              lineColor: (lineData, index, id) {
                return id == 'Line 1'
                    ? Colors.blue
                    : id == 'Line 2'
                    ? Colors.amber
                    : Colors.green;
              },
              pointColor: (lineData, index, id) {
                return id == 'Line 1'
                    ? Colors.blue.shade900
                    : id == 'Line 2'
                    ? Colors.amber.shade900
                    : Colors.green.shade900;
              },
              data: rawData,
              includePoints: true,
              includeArea: true,
            ),
          ),
        ],
      ),
    );
  }
}


