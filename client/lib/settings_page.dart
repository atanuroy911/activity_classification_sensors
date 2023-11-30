import 'package:flutter/material.dart';
import 'package:settings_ui/settings_ui.dart';
import 'package:shared_preferences/shared_preferences.dart';

class SettingsPage extends StatefulWidget {
  const SettingsPage({Key? key}) : super(key: key);

  @override
  _SettingsPageState createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  TextEditingController urlController = TextEditingController();
  TextEditingController portController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _loadSavedSettings();
  }

  Future<void> _loadSavedSettings() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    setState(() {
      urlController.text = prefs.getString('url') ?? '';
      portController.text = prefs.getString('port') ?? '';
    });
  }

  Future<void> _saveSettings() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    print(urlController.text);
    await prefs.setString('url', urlController.text);
    await prefs.setString('port', portController.text);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // appBar: AppBar(title: Text('Settings')),
      body: SettingsList(
        sections: [
          SettingsSection(
            title: const Text('Connection Settings'),
            tiles: [
              SettingsTile(
                title: const Text('IP Address (without http/https)'),
                description: Text(urlController.text),
                onPressed: (BuildContext context) async {
                  final result = await showDialog(
                    context: context,
                    builder: (_) =>
                        _buildInputDialog('URL', urlController.text),
                  );
                  if (result != null) {
                    setState(() {
                      print(result);
                      urlController.text = result;
                    });
                    _saveSettings();
                  }
                },
              ),
              SettingsTile(
                title: const Text('Port'),
                description: Text(portController.text),
                onPressed: (BuildContext context) async {
                  final result = await showDialog(
                    context: context,
                    builder: (_) =>
                        _buildInputDialog('Port', portController.text),
                  );
                  if (result != null) {
                    setState(() {
                      print(result);
                      portController.text = result;
                    });
                    _saveSettings();
                  }
                },
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildInputDialog(String title, String initialValue) {
    TextEditingController inputController = TextEditingController(
        text: initialValue); // Create a controller for the input field

    return AlertDialog(
      title: Text('Enter $title'),
      content: TextFormField(
        controller: inputController, // Set the controller for the input field
        autofocus: true,
        decoration: InputDecoration(labelText: title),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: Text('Cancel'),
        ),
        TextButton(
          onPressed: () {
            Navigator.pop(context, initialValue); // Return the original value
          },
          child: Text('Use Default'),
        ),
        ElevatedButton(
          onPressed: () {
            Navigator.pop(
                context, inputController.text); // Return the updated text
          },
          child: Text('Apply'),
        ),
      ],
    );
  }
}
