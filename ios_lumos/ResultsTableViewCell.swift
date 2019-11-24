//
//  ResultsTableViewCell.swift
//  lumos ios
//
//  Created by Shalin Shah on 7/20/17.
//  Copyright © 2017 Shalin Shah. All rights reserved.
//

import UIKit

class ResultsTableViewCell: UITableViewCell {

    @IBOutlet weak var resultsTableImage: UIImageView!
    @IBOutlet weak var noImageStatLabel: UILabel!
    @IBOutlet weak var imageStatLabel: UILabel!
    @IBOutlet weak var resultDescription: UILabel!

    override func awakeFromNib() {
        super.awakeFromNib()
        // Initialization code
        
        
        // Text style for "resultDescription"
        resultDescription.clipsToBounds = true
        resultDescription.alpha = 1
        resultDescription.font = UIFont(name: "AvenirNext-DemiBold", size: 20)
        resultDescription.textColor = UIColor(red: 1.00, green: 1.00, blue: 1.00, alpha: 1)
        
        imageStatLabel.isHidden = true
        
        
        // Text style for "noImageStatLabel"
        noImageStatLabel.clipsToBounds = true
        noImageStatLabel.alpha = 1
        noImageStatLabel.font = UIFont(name: "AvenirNext-DemiBold", size: 50)
        noImageStatLabel.textColor = UIColor(red: 1.00, green: 1.00, blue: 1.00, alpha: 1)
        
    }

    override func setSelected(_ selected: Bool, animated: Bool) {
        super.setSelected(selected, animated: animated)

        // Configure the view for the selected state
    }

}
